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
    rm -rf .git/
    git config --global user.email "travis"
    git config --global user.name "travis"
    git init
    git add index.html
    git add .nojekyll
    git add doc
    echo "Committing"
    git commit -m init
    # Make sure to make the output quiet, or else the API token will leak!
    # This works because the API key can replace your password.
    echo "Pushing"
    git push -f -q "https://${GITHUB_API_KEY2}@${GH_REF}" master:gh-pages > /dev/null 2>&1 && echo "Pushed"
fi
echo "Deploy script ending"