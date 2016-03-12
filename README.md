

## Build documentation

1. Navigate to doc directory
2. `make rst`
3. `make html`
4. `make latexpdf` (if you also want to pdf output)
5. Documentation should now be found in the `_build` directory underneath ``doc``. The top level ``index.html`` file will redirect to the appropriate place, just open it up.