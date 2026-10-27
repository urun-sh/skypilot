const nextJest = require('next/jest');

const createJestConfig = nextJest({
  // Provide the path to your Next.js app to load next.config.js and .env files in your test environment
  dir: './',
});

// Add any custom config to be passed to Jest
const customJestConfig = {
  testEnvironment: 'jsdom',
  setupFilesAfterEnv: ['<rootDir>/jest.setup.js'],
  verbose: true,
  silent: false,
  injectGlobals: true,
  testMatch: ['**/__tests__/**/*.[jt]s?(x)', '**/?(*.)+(spec|test).[jt]s?(x)'],
  moduleNameMapper: {
    '^@/(.*)$': '<rootDir>/src/$1',
  },
  transform: {
    '^.+\\.(js|jsx|ts|tsx)$': 'babel-jest',
  },
  transformIgnorePatterns: [
    '/node_modules/',
    '^.+\\.module\\.(css|sass|scss)$',
  ],
  testPathIgnorePatterns: ['<rootDir>/node_modules/', '<rootDir>/.next/'],
};

// createJestConfig is exported this way to ensure that next/jest can load the Next.js config which is async
const createNextJestConfig = createJestConfig(customJestConfig);
module.exports = async (...args) => {
  const config = await createNextJestConfig(...args);
  // next/jest appends its own transformIgnorePatterns defaults, and jest
  // ignores a file when ANY pattern matches, so its blanket node_modules
  // ignore would keep d3-color untransformed even with an allowlist
  // appended to customJestConfig. Replace the merged list instead:
  // d3-color >= 3 is ESM-only ("main": "src/index.js"), so the jsdom
  // suites need it compiled to CJS, the same treatment next/jest gives
  // its own allowlisted packages. This repo has a flat npm node_modules
  // (no .pnpm store) and geist is not a dependency, so next's pnpm/geist
  // patterns have nothing left to cover here.
  return {
    ...config,
    transformIgnorePatterns: [
      '/node_modules/(?!d3-color/)',
      '^.+\\.module\\.(css|sass|scss)$',
    ],
  };
};
