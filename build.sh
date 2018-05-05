#!/bin/sh

if [ "$COMPILER_NAME" = "g++" ]; then
  make coverage
elif [ "$COMPILER_NAME" = "clang++" ]; then
  make clang_all
else
  echo "Invalid compiler"
  exit 1
fi
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "Build failed" >&2
  exit "$rc"
fi
for run in {1..5}
do
  make check
  rc=$?
  if [ "$rc" -eq 0 ]; then
    break
  fi
done
if [ "$rc" -ne 0 ]; then
  echo "Test failed" >&2
  exit "$rc"
fi
if [ "$COMPILER_NAME" = "g++" ]; then
  make report
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "GNU Coverage report failed" >&2
    exit "$rc"
  fi
fi