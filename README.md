## UCCL Website

### Deploy and Test on a Local Mac

```shell
brew update
brew install node pnpm
pnpm install
pnpm run dev
```

### Local fallback for not-yet-merged GitHub-hosted assets

If a new post references an asset that will only exist on GitHub after the PR is merged, mirror that file under `public/assets/...` and run locally with:

```shell
LOCAL_GITHUB_ASSET_FALLBACK=1 pnpm run dev --host 0.0.0.0
```

The default build and deployed site keep using the original `raw.githubusercontent.com/.../main/assets/...` URLs.
