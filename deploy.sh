#!/usr/bin/env bash
if [ "$TRAVIS_PULL_REQUEST" != "false" -o "$TRAVIS_BRANCH" != "master" ];
    then exit 0;
fi
if [ -n "$GITHUB_API_KEY" ]; then
    echo "Github key found. Building documentation."
    cd "$TRAVIS_BUILD_DIR"/doc
    make clean
    make rst
    make html
    cd "$TRAVIS_BUILD_DIR"
    git config --global user.email "travis"
    git config --global user.name "travis"
    git init
    git checkout -b gh-pages
    git add index.html
    git add .nojekyll
    git add doc
    git commit -m init
    # Make sure to make the output quiet, or else the API token will leak!
    # This works because the API key can replace your password.
    git push -f -q https://$GITHUB_API_KEY@github.com/dessn/sn-doc gh-pages &2>/dev/null
fi
echo "Deploy script ending"