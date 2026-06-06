# build stage
FROM node:24-slim AS build

WORKDIR /app

COPY package.json yarn.lock* ./
RUN yarn install --frozen-lockfile

COPY . .
RUN yarn build

# dev stage
FROM node:24-slim AS dev
WORKDIR /app
COPY package.json yarn.lock* ./
RUN yarn install
COPY . .
EXPOSE 3000
CMD ["yarn", "start"]

# prod stage
FROM caddy:2-alpine AS prod

COPY --from=build /app/build /usr/share/caddy

EXPOSE 3000
