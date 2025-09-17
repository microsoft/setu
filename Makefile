# -*- makefile -*-
# setu Project Makefile
# This Makefile provides a clean interface to build scripts

# Default shell
SHELL := bash

# Build script directory
SCRIPTS_DIR := $(CURDIR)/scripts/dev

# Environment variables
export USE_RAMDISK ?= 1
export SETU_IS_CI_CONTEXT ?= 0

# Default target
.DEFAULT_GOAL := help

# === HELP ===
.PHONY: help
help: ## show this help message
	@echo "setu Project Build System"
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@awk 'BEGIN {FS = ":.*##"; printf "\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

# === LINTING ===
##@ Linting

.PHONY: lint lint/black lint/isort lint/autoflake lint/pyright lint/cpplint lint/clang-format lint/codespell lint/shellcheck lint/cmake-lint
lint: ## lint all code
	$(SCRIPTS_DIR)/lint.sh all

lint/black: ## check python style with black
	$(SCRIPTS_DIR)/lint.sh black

lint/isort: ## check python style with isort
	$(SCRIPTS_DIR)/lint.sh isort

lint/autoflake: ## check for unused python imports
	$(SCRIPTS_DIR)/lint.sh autoflake

lint/pyright: ## run python type checking
	$(SCRIPTS_DIR)/lint.sh pyright

lint/cpplint: ## run C++ style checks with cpplint
	$(SCRIPTS_DIR)/lint.sh cpplint

lint/clang-format: ## check C++ format with clang-format
	$(SCRIPTS_DIR)/lint.sh clang-format

lint/codespell: ## check for common misspellings
	$(SCRIPTS_DIR)/lint.sh codespell

lint/shellcheck: ## check shell scripts with shellcheck
	$(SCRIPTS_DIR)/lint.sh shellcheck

lint/cmake-lint: ## check CMake files with cmake-lint
	$(SCRIPTS_DIR)/lint.sh cmake-lint

# === FORMATTING ===
##@ Formatting

.PHONY: format format/black format/isort format/autoflake format/clang format/shfmt format/cmake-format format/yaml
format: ## format all code
	$(SCRIPTS_DIR)/format.sh all

format/black: ## format python code with black
	$(SCRIPTS_DIR)/format.sh black

format/isort: ## format python imports with isort
	$(SCRIPTS_DIR)/format.sh isort

format/autoflake: ## remove unused python imports
	$(SCRIPTS_DIR)/format.sh autoflake

format/clang: ## format C++ code with clang-format
	$(SCRIPTS_DIR)/format.sh clang

format/shfmt: ## format shell scripts with shfmt
	$(SCRIPTS_DIR)/format.sh shfmt

format/cmake-format: ## format CMake files with cmake-format
	$(SCRIPTS_DIR)/format.sh cmake-format

format/yaml: ## format YAML files with yamlfmt
	$(SCRIPTS_DIR)/format.sh yaml

# === BUILDING ===
##@ Building

# User-facing build targets
.PHONY: build build/test build/wheel build/editable
build: ## intelligent build - automatically selects optimal strategy
	$(SCRIPTS_DIR)/build.sh smart

build/test: ## intelligent test build - automatically selects optimal test strategy
	$(SCRIPTS_DIR)/build.sh smart-test

# Advanced build targets (for power users and CI)
.PHONY: build/native build/native_incremental build/native_test build/native_test_full build/native_test_incremental
build/native: ## build native extension (full build)
	$(SCRIPTS_DIR)/build.sh native

build/native_incremental: ## build native extension (incremental)
	$(SCRIPTS_DIR)/build.sh native-incremental

build/native_test: ## intelligent native test build - automatically selects optimal test strategy
	$(SCRIPTS_DIR)/build.sh native-test

build/native_test_incremental: ## build native tests (incremental)
	$(SCRIPTS_DIR)/build.sh native-test-incremental

build/wheel: ## build python wheel and sdist
	$(SCRIPTS_DIR)/wheel.sh

build/editable: ## build project (install editable package) - legacy
	$(SCRIPTS_DIR)/build.sh editable

# === TESTING ===
##@ Testing

.PHONY: test test/unit test/integration test/functional test/pyunit test/pyintegration test/ctest test/ctest_incremental test/performance test/correctness test/unit-failed-only test/pyunit-failed-only test/ctest-failed-only
test: ## run all tests
	$(SCRIPTS_DIR)/test.sh all

test/unit: ## run all unit tests (py + C++)
	$(SCRIPTS_DIR)/test.sh unit

test/unit-failed-only: ## rerun only failed tests from last unit test run (py + C++)
	$(SCRIPTS_DIR)/test.sh unit-failed-only

test/integration: ## run all integration tests (py)
	$(SCRIPTS_DIR)/test.sh integration

test/functional: ## run all functional tests
	$(SCRIPTS_DIR)/test.sh functional

test/pyunit: ## run python unit tests with reports
	$(SCRIPTS_DIR)/test.sh pyunit

test/pyunit-failed-only: ## rerun only failed Python unit tests from last run
	$(SCRIPTS_DIR)/test.sh pyunit-failed-only

test/pyintegration: ## run python integration tests with reports
	$(SCRIPTS_DIR)/test.sh pyintegration

test/ctest: ## run C++ tests (ctest) with reports
	$(SCRIPTS_DIR)/test.sh ctest

test/ctest-failed-only: ## rerun only failed C++ tests from last ctest run
	$(SCRIPTS_DIR)/test.sh ctest-failed-only


# === SETUP ===
##@ Setup

.PHONY: setup/environment setup/update-environment setup/check setup/activate

setup/environment: ## create conda/mamba development environment
	$(SCRIPTS_DIR)/setup.sh environment

setup/update-environment: ## update conda/mamba development environment
	$(SCRIPTS_DIR)/setup.sh update-environment

setup/check: ## check system and python environment
	$(SCRIPTS_DIR)/setup.sh check

setup/activate: ## show command to activate environment
	@echo "To activate the environment, run:"
	@if [ -d "./env" ]; then \
		echo "  conda activate ./env"; \
	else \
		echo "  Environment not found. Run 'make setup/environment' first."; \
	fi

# === DOCUMENTATION ===
##@ Documentation

.PHONY: docs docs/build docs/clean docs/serve
docs: ## build documentation
	$(SCRIPTS_DIR)/docs.sh all

docs/build: ## build documentation using sphinx
	$(SCRIPTS_DIR)/docs.sh build

docs/clean: ## clean documentation build artifacts
	$(SCRIPTS_DIR)/docs.sh clean

docs/serve: ## serve documentation locally (default port 8000)
	$(SCRIPTS_DIR)/docs.sh serve

# === DEV CONTAINERS ===
##@ Dev Containers

.PHONY: dev_container/start dev_container/stop dev_container/attach
dev_container/start: ## start development container (requires USERNAME)
	$(MAKE) -C $(CURDIR)/docker/containers/dev start USERNAME=$(USERNAME)

dev_container/stop: ## stop development container (requires USERNAME)
	$(MAKE) -C $(CURDIR)/docker/containers/dev stop USERNAME=$(USERNAME)

dev_container/attach: ## attach to running development container (requires USERNAME)
	$(MAKE) -C $(CURDIR)/docker/containers/dev attach USERNAME=$(USERNAME)

# === UTILITIES ===
##@ Utilities

.PHONY: clean ccache/stats ccache/clear
clean: ## clean build artifacts, logs, caches
	$(SCRIPTS_DIR)/build.sh clean

clean/force: ## clean build artifacts, logs, caches
	$(SCRIPTS_DIR)/build.sh clean-force

ccache/stats: ## show ccache statistics
	$(SCRIPTS_DIR)/build.sh ccache-stats

ccache/clear: ## clear ccache
	$(SCRIPTS_DIR)/build.sh ccache-clear
