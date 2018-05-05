#!/bin/sh

if [ "$COMPILER_NAME" = "g++" ]; then
  make coverage
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "GCC build failed" >&2
    exit "$rc"
  fi
  for run in {1.."$TRIES"}
  do
    make check
    rc=$?
    if [ "$rc" -eq 0 ]; then
      break
    fi
  done
  if [ "$rc" -ne 0 ]; then
    echo "GCC test failed" >&2
    exit "$rc"
  fi
  make report
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "GNU Coverage report failed" >&2
    exit "$rc"
  fi
elif [ "$COMPILER_NAME" = "clang++" ]; then
  make clang_all
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "Clang build failed" >&2
    exit "$rc"
  fi
  for run in {1.."$TRIES"}
  do
    make check
    rc=$?
    if [ "$rc" -eq 0 ]; then
      break
    fi
  done
  if [ "$rc" -ne 0 ]; then
    echo "Clang test failed" >&2
    exit "$rc"
  fi
else
  echo 'Invalid compiler' >&2
  exit 1
fi