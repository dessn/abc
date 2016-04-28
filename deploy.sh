#!/usr/bin/env bash
if [ -n "$GITHUB_API_KEY" ]; then
    cd "$TRAVIS_BUILD_DIR"/doc
    echo `pwd`
    echo `ls`
    make clean
    make rst
    make html
    cd "$TRAVIS_BUILD_DIR"
    #git init
    #git checkout -b gh-pages
    #git add .
    #git -c user.name='travis' -c user.email='travis' commit -m init
    # Make sure to make the output quiet, or else the API token will leak!
    # This works because the API key can replace your password.
    #git push -f -q https://<me>:$GITHUB_API_KEY@github.com/dessn/abc gh-pages &2>/dev/null
  fi
echo "DOOM"