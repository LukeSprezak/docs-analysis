# build stage
FROM node:24-slim AS build

WORKDIR /app

# Corepack reads `packageManager` from package.json and fetches exactly that yarn, so the
# image and the host cannot drift on the package manager version. The fallback covers base
# images that no longer bundle corepack (Node dropped it from the distribution in 25).
ENV COREPACK_ENABLE_DOWNLOAD_PROMPT=0
RUN corepack enable || (npm install -g corepack && corepack enable)

COPY package.json yarn.lock .yarnrc.yml ./

# --immutable is Yarn 4's spelling of --frozen-lockfile: the install fails (YN0028) instead
# of quietly rewriting yarn.lock.
RUN yarn install --immutable

COPY . .
# `yarn build` is `tsc --noEmit && vite build`, so the dev dependencies installed above are
# required here — do not narrow the install to production dependencies.
RUN yarn build

# dev stage
FROM node:24-slim AS dev
WORKDIR /app

ENV COREPACK_ENABLE_DOWNLOAD_PROMPT=0
RUN corepack enable || (npm install -g corepack && corepack enable)

COPY package.json yarn.lock .yarnrc.yml ./
RUN yarn install --immutable

COPY . .
EXPOSE 3000
CMD ["yarn", "start"]

# prod stage
FROM caddy:2-alpine AS prod

# Vite's outDir (client/vite.config.ts), not CRA's build/.
COPY --from=build /app/dist /usr/share/caddy

# Matches the site address in docker/caddy/Caddyfile, which compose mounts at
# /etc/caddy/Caddyfile. Without that mount Caddy falls back to its own default on :80.
EXPOSE 3000
