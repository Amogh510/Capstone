const fs = require('fs');
const path = require('path');

/**
 * Normalizes a file path to use forward slashes and be absolute.
 * @param {string} filePath - The file path to normalize.
 * @returns {string} The normalized, absolute file path.
 */
function normalizePath(filePath) {
  return path.resolve(filePath).replace(/\\/g, '/');
}

/**
 * Writes data to a JSON file.
 * @param {string} filePath - The path to the output file.
 * @param {object} data - The JSON data to write.
 */
function writeJsonFile(filePath, data) {
  try {
    fs.writeFileSync(filePath, JSON.stringify(data, null, 2));
    console.log(`Successfully wrote to ${filePath}`);
  } catch (error) {
    console.error(`Error writing to ${filePath}:`, error);
  }
}

module.exports = {
  normalizePath,
  writeJsonFile,
}; 