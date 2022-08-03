import nox


def install_flit_dev_deps(session):
    session.install("flit")
    session.run("flit", "install", "--deps", "develop")


@nox.session(python=["3.8"])
def tests(session):
    install_flit_dev_deps(session)
    session.run("pytest", "--cov=movinets_helper", "--cov-report=xml", "tests")


@nox.session
def lint(session):
    install_flit_dev_deps(session)
    session.run("isort", "movinets_helper")
    session.run("black", "--check", "movinets_helper")
    session.run("mypy", "movinets_helper")
    session.run("make", "docs")
