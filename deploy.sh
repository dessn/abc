#!/usr/bin/env bash
if [ -n "$GITHUB_API_KEY" ]; then
    echo "Github key found. Building documentation."
    cd "$TRAVIS_BUILD_DIR"/doc
    make clean
    make rst
    make html
    cd "$TRAVIS_BUILD_DIR"
    git init
    git checkout -b gh-pages
    git config --global user.email "travis"
    git config --global user.email "travis"
    git add index.html
    git add .nojekyll
    git add doc
    git commit -m init
    # Make sure to make the output quiet, or else the API token will leak!
    # This works because the API key can replace your password.
    git push -f -q https://$GITHUB_API_KEY@github.com/dessn/sn-doc gh-pages &2>/dev/null
fi
echo "Deploy script ending"