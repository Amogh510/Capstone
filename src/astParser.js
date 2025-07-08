const fs = require('fs');
const parser = require('@babel/parser');
const path = require('path');

/**
 * Parses a file and returns its AST.
 * @param {string} filePath - The path to the file to parse.
 * @returns {object|null} The AST object, or null if parsing fails.
 */
function parseFile(filePath) {
  try {
    const code = fs.readFileSync(filePath, 'utf-8');
    const fileExt = path.extname(filePath).toLowerCase();
    
    // Don't parse non-JavaScript/TypeScript files
    if (!['.js', '.jsx', '.ts', '.tsx'].includes(fileExt)) {
      return null;
    }
    
    return parser.parse(code, {
      sourceType: 'module',
      plugins: [
        'jsx',
        'typescript',
        'decorators-legacy',
        'classProperties',
        'objectRestSpread'
      ],
    });
  } catch (error) {
    console.error(`Error parsing ${filePath}:`, error);
    return null;
  }
}

module.exports = {
  parseFile,
}; 