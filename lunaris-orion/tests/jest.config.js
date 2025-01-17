module.exports = {
    testEnvironment: 'node',
    verbose: true,
    collectCoverage: true,
    coverageDirectory: 'coverage',
    coverageReporters: ['text', 'lcov'],
    testMatch: ['**/tests/**/*.test.js'],
    setupFilesAfterEnv: ['<rootDir>/tests/setup.js'],
    moduleNameMapper: {
        '^@/(.*)$': '<rootDir>/src/$1'
    }
}; 