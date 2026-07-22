"""Gym-Anything OpenEnv client adapter.

Wraps gym-anything's GymAnythingEnv to provide:
- reset_async() / step_async() matching Fleet env interface
- Screenshot observations as base64 image_url blocks
- Programmatic verifier rewards (0-100 normalized to 0-1)
- computer_use MCP tool definitions compatible with Qwen VL

Usage:
    from envs.gym_anything_env import GymAnythingEnvClient

    env = GymAnythingEnvClient(
        env_dir="/path/to/gym-anything/benchmarks/cua_world/environments/blender3d_env",
        task_id="add_sphere_to_scene",
    )
    obs = await env.reset_async()
    obs, reward, done, info = await env.step_async(action)
"""

import asyncio
import base64
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# MCP-compatible computer_use tool definition
COMPUTER_USE_TOOL = {
    "type": "function",
    "function": {
        "name": "computer",
        "description": (
            "Use a mouse and keyboard to interact with a computer, and take screenshots.\n"
            "* This is an interface to a desktop GUI.\n"
            "* The screen's resolution is 1000x1000.\n"
            "* Coordinates use a [0, 1000] grid. (0,0) is top-left, (999,999) is bottom-right.\n"
            "* Click the center of elements, not their edges."
        ),
        "parameters": {
            "type": "object",
            "required": ["action"],
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "key", "type", "mouse_move", "click", "left_click",
                        "drag", "right_click", "double_click", "triple_click",
                        "scroll", "wait", "screenshot",
                    ],
                },
                "keys": {"type": "array"},
                "text": {"type": "string"},
                "coordinate": {"type": "array"},
                "coordinate2": {"type": "array"},
                "pixels": {"type": "number"},
                "time": {"type": "number"},
            },
        },
    },
}


class GymAnythingEnvClient:
    """OpenEnv-compatible client for gym-anything desktop environments."""

    def __init__(
        self,
        env_dir: str,
        task_id: Optional[str] = None,
        use_cache: bool = True,
        cache_level: str = "post_start",
        max_steps: int = 50,
    ):
        self.env_dir = env_dir
        self.task_id = task_id
        self.use_cache = use_cache
        self.cache_level = cache_level
        self.max_steps = max_steps

        self.ga_env = None
        self.screen_width = 1920
        self.screen_height = 1080
        self._step_count = 0

    async def reset_async(self) -> Dict[str, Any]:
        """Reset environment, return initial observation with tools."""
        from gym_anything import from_config

        if self.ga_env:
            self.ga_env.close()

        self.ga_env = from_config(Path(self.env_dir), task_id=self.task_id)

        screen_spec = next(
            (o for o in self.ga_env.env_spec.observation if o.type == "rgb_screen"),
            None,
        )
        if screen_spec and screen_spec.resolution:
            self.screen_width, self.screen_height = screen_spec.resolution

        obs = await asyncio.to_thread(
            self.ga_env.reset,
            use_cache=self.use_cache,
            cache_level=self.cache_level,
        )

        self._step_count = 0
        screenshot = self._obs_to_screenshot(obs)

        result = {"tools": [COMPUTER_USE_TOOL]}
        if screenshot:
            result["initial_screenshot"] = [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot}"}}
            ]

        return result

    async def step_async(self, action: Dict[str, Any]) -> Tuple[Dict, float, bool, Dict]:
        """Execute action, return (obs, reward, done, info)."""
        self._step_count += 1
        is_done = action.get("done", False)

        tool_name = action.get("tool", "")
        params = action.get("params", {})
        action_type = params.get("action", "")

        if tool_name == "computer" and action_type:
            ga_actions = self._convert_action(params)
            obs, reward, done, info = await asyncio.to_thread(
                self.ga_env.step, ga_actions, mark_done=is_done,
            )
        elif is_done:
            obs, reward, done, info = await asyncio.to_thread(
                self.ga_env.step, [{"action": "screenshot"}], mark_done=True,
            )
        else:
            obs = self.ga_env.capture_observation()
            reward, done, info = 0.0, False, {}

        screenshot = self._obs_to_screenshot(obs)
        observation = {}
        if screenshot:
            observation["observation"] = [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot}"}}
            ]

        # Normalize reward from 0-100 to 0-1
        if done and "verifier" in info:
            reward = info["verifier"].get("score", 0) / 100.0

        return observation, reward, done or is_done, info

    def close(self):
        if self.ga_env:
            self.ga_env.close()
            self.ga_env = None

    def _obs_to_screenshot(self, obs: Dict[str, Any]) -> Optional[str]:
        screen = obs.get("screen", {})
        if "png_b64" in screen:
            return screen["png_b64"]
        path = screen.get("path")
        if path and os.path.exists(path):
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode("ascii")
        return None

    def _scale(self, x: int, y: int) -> Tuple[int, int]:
        return int(x / 1000 * self.screen_width), int(y / 1000 * self.screen_height)

    def _convert_action(self, params: Dict[str, Any]) -> List[Dict]:
        action_type = params.get("action", "")
        coord = params.get("coordinate", [500, 500])

        if action_type == "screenshot":
            return [{"action": "screenshot"}]
        if action_type == "wait":
            return [{"action": "wait", "time": params.get("time", 1.0)}]
        if action_type == "key":
            keys = params.get("keys", [])
            return [{"keyboard": {"keys": keys if isinstance(keys, list) else [keys]}}]
        if action_type == "type":
            actions = []
            if params.get("clear"):
                actions.append({"keyboard": {"keys": ["ctrl", "a"]}})
            actions.append({"keyboard": {"text": params.get("text", "")}})
            if params.get("enter"):
                actions.append({"keyboard": {"keys": ["Return"]}})
            return actions
        if action_type in ("click", "left_click"):
            x, y = self._scale(coord[0], coord[1])
            return [{"mouse": {"left_click": [x, y]}}]
        if action_type == "right_click":
            x, y = self._scale(coord[0], coord[1])
            return [{"mouse": {"right_click": [x, y]}}]
        if action_type == "double_click":
            x, y = self._scale(coord[0], coord[1])
            return [{"mouse": {"double_click": [x, y]}}]
        if action_type == "triple_click":
            x, y = self._scale(coord[0], coord[1])
            return [{"mouse": {"triple_click": [x, y]}}]
        if action_type == "mouse_move":
            x, y = self._scale(coord[0], coord[1])
            return [{"mouse": {"move": [x, y]}}]
        if action_type in ("drag", "left_click_drag"):
            coord2 = params.get("coordinate2", coord)
            x1, y1 = self._scale(coord[0], coord[1])
            x2, y2 = self._scale(coord2[0], coord2[1])
            return [{"mouse": {"left_click_drag": [[x1, y1], [x2, y2]]}}]
        if action_type == "scroll":
            actions = []
            if "coordinate" in params:
                x, y = self._scale(coord[0], coord[1])
                actions.append({"mouse": {"move": [x, y]}})
            actions.append({"mouse": {"scroll": int(params.get("pixels", 0))}})
            return actions

        return [{"action": "screenshot"}]
