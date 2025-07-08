const fs = require('fs');
const parser = require('@babel/parser');

/**
 * Parses a file and returns its AST.
 * @param {string} filePath - The path to the file to parse.
 * @returns {object|null} The AST object, or null if parsing fails.
 */
function parseFile(filePath) {
  try {
    const code = fs.readFileSync(filePath, 'utf-8');
    return parser.parse(code, {
      sourceType: 'module',
      plugins: ['jsx'],
    });
  } catch (error) {
    console.error(`Error parsing ${filePath}:`, error);
    return null;
  }
}

module.exports = {
  parseFile,
}; 