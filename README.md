#Bayesian Supernova Cosmology framework

####For the DES Supernova Working Group

[![Build Status](https://travis-ci.org/dessn/abc.svg?style=flat&branch=master)](https://travis-ci.org/dessn/abc)
[![Coverage Status](https://coveralls.io/repos/github/dessn/abc/badge.svg?branch=master)](https://coveralls.io/github/dessn/abc?branch=master)
[![License](http://img.shields.io/badge/license-MIT-blue.svg?style=flat)](https://github.com/dessn/abc/blob/master/LICENSE)

## Manually building documentation

[**Documentation available online here.**](https://dessn.github.io/sn-doc)

If you wish to generate the documentation yourself (mainly if you wish to generate
a PDF file, which Travis does not do), you will need to follow the below steps.

1. Navigate to doc directory
2. Execute `make rst`
3. Execute `make latexpdf` (if you want PDF output)
4. Execute make html` (if you want HTML output)
5. Documentation should now be found in the `_build` directory underneath doc`.
   The top level `index.html` file will redirect to the appropriate place, just open it up.