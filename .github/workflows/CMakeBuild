workflow "compile a cmake project" {
  on = "push"
  resolves = "build"
}

action "build" {
  uses = "popperized/cmake@master"
  args = "install"
  env = {
    CMAKE_PROJECT_DIR = "./",
    CMAKE_FLAGS = "-DCMAKE_INSTALL_PREFIX:PATH=$GITHUB_WORKSPACE/install",
    CMAKE_CLEAN = 1
  }
}

workflow "Unit Test" {
  on = "push"
  resolves = ["action-gtest"]
}

action "action-gtest" {
  uses = "CyberZHG/github-action-gtest@master"
  args = "-d test -e testing-all"
}
