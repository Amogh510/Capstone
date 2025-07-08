const glob = require('glob');
const path = require('path');
const { projectRoot, fileExtensions, ignorePatterns } = require('./config');
const { normalizePath } = require('./utils/fileUtils');

/**
 * Recursively finds all relevant files in the project directory.
 * @returns {string[]} An array of absolute file paths.
 */
function walk() {
  const pattern = `**/*{${fileExtensions.join(',')}}`;
  const options = {
    cwd: projectRoot,
    nodir: true,
    ignore: ignorePatterns,
  };

  try {
    const files = glob.sync(pattern, options);
    return files.map((file) => normalizePath(path.join(projectRoot, file)));
  } catch (error) {
    console.error('Error walking files:', error);
    return [];
  }
}

module.exports = {
  walk,
}; 