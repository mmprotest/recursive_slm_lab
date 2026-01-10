from typer.testing import CliRunner

from recursive_slm_lab.cli import app


def test_self_improve_help_includes_log_level() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["self-improve", "--help"])

    assert result.exit_code == 0
    assert "--log-level" in result.output
