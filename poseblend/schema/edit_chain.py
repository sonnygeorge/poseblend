import os

from pydantic import BaseModel


class CheckResult(BaseModel):
    checked: str
    did_pass: bool
    confidence: float
    reasoning: str


class AttemptedEdit(BaseModel):
    before_img_path: os.PathLike
    after_img_path: os.PathLike
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
    starting_render_path: os.PathLike
    edits: list[list[AttemptedEdit]]
    final_img_path: os.PathLike | None
    fail_reason: str | None

    @property
    def was_failure(self) -> bool:
        has_fail_reason = self.fail_reason is not None
        assert not (has_fail_reason and self.final_img_path is not None)
        return has_fail_reason

    @property
    def was_success(self) -> bool:
        has_final_img = self.final_img_path is not None
        assert not (has_final_img and self.was_failure)
        return has_final_img
