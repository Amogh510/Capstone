const path = require('path');

module.exports = {
  // Define the root directory of the project to be analyzed
  projectRoot: path.resolve(__dirname, '..', '/Users/aneesh/Authify/ExampleBrand1/client/src'), // IMPORTANT: Change this to the actual project root

  // File extensions to include in the analysis
  fileExtensions: ['.js', '.jsx', '.css', '.scss', '.sass'],

  // Patterns to ignore during file traversal
  ignorePatterns: [
    'node_modules/',
    'build/',
    'dist/',
    'public/',
    '*.test.js',
    '*.spec.js',
  ],

  // Path aliases for resolving module imports
  // Example: { '@components': './src/components' }
  alias: {
    // Add your webpack/jsconfig path aliases here
  },

  // List of JSX tags to extract for the Knowledge Graph
  jsxTagAllowList: ['button', 'input', 'form', 'select', 'textarea', 'label', 'a'],

  // Output directory for the generated JSON files
  outputDir: path.resolve(__dirname, 'output'),
}; 