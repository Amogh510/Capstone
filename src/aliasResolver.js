const path = require('path');
const { alias, projectRoot } = require('./config');
const { normalizePath } = require('./utils/fileUtils');

/**
 * Resolves an aliased import path to an absolute path.
 * @param {string} importPath - The import path to resolve.
 * @param {string} currentFileDir - The directory of the file containing the import.
 * @returns {string} The resolved, absolute path.
 */
function resolveAlias(importPath, currentFileDir) {
  const aliasMatch = Object.keys(alias).find((key) =>
    importPath.startsWith(key)
  );

  if (aliasMatch) {
    const aliasPath = alias[aliasMatch];
    const resolvedPath = path.join(
      projectRoot,
      aliasPath,
      importPath.substring(aliasMatch.length)
    );
    return normalizePath(resolvedPath);
  }

  return normalizePath(path.resolve(currentFileDir, importPath));
}

module.exports = {
  resolveAlias,
}; 