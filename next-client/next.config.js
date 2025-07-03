// @ts-check

/** @type {import('next').NextConfig} */
const nextConfig = {
  /* config options here */
  output: "standalone",
  experimental: {
    serverActions: {
      bodySizeLimit: "10mb", // TODO: make configurable via env var when Next.js types support it
    },
  },
};

module.exports = nextConfig;
