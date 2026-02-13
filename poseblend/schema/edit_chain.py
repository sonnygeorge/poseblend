from pathlib import Path

from pydantic import BaseModel


class CheckResult(BaseModel):
    checked: str
    did_pass: bool
    confidence: float
    reasoning: str


class AttemptedEdit(BaseModel):
    before_img_path: Path
    after_img_path: Path
    check_results: list[CheckResult]

    @property
    def was_failure(self) -> bool:
        if len(self.check_results) == 0:
            return False
        last_result_failed = not self.check_results[-1].did_pass
        return last_result_failed

    @property
    def failure_reason(self) -> str | None:
        if not self.was_failure:
            return None
        return self.check_results[-1].reasoning


class EditChain(BaseModel):
    starting_render_path: Path
    edits: list[list[AttemptedEdit]]
    final_img_path: Path | None
    fail_reason: str | None

    @property
    def was_failure(self) -> bool:
        has_fail_reason = self.fail_reason is not None
        if has_fail_reason and self.final_img_path is not None:
            msg = "EditChain cannot have both a fail_reason and a final_img_path"
            raise ValueError(msg)
        return has_fail_reason

    @property
    def was_success(self) -> bool:
        has_final_img = self.final_img_path is not None
        if has_final_img and self.fail_reason is not None:
            msg = "EditChain cannot have both a final_img_path and a fail_reason"
            raise ValueError(msg)
        return has_final_img
