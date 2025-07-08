const path = require('path');
const fs = require('fs');
const { alias, projectRoot } = require('./config');
const { normalizePath } = require('./utils/fileUtils');

/**
 * Resolves an aliased import path to an absolute path.
 * @param {string} importPath - The import path to resolve.
 * @param {string} currentFileDir - The directory of the file containing the import.
 * @returns {string} The resolved, absolute path.
 */
function resolveAlias(importPath, currentFileDir) {
  // Handle relative imports (starting with ./ or ../)
  if (importPath.startsWith('./') || importPath.startsWith('../')) {
    const resolvedPath = path.resolve(currentFileDir, importPath);
    
    // Try to find the actual file with extensions
    const possibleExtensions = ['.js', '.jsx', '.ts', '.tsx', '.json'];
    let finalPath = resolvedPath;
    
    // If the resolved path doesn't have an extension, try to find it
    if (!path.extname(resolvedPath)) {
      for (const ext of possibleExtensions) {
        const pathWithExt = resolvedPath + ext;
        if (fs.existsSync(pathWithExt)) {
          finalPath = pathWithExt;
          break;
        }
      }
    }
    
    return normalizePath(finalPath);
  }

  // Handle absolute imports (starting with /)
  if (importPath.startsWith('/')) {
    const resolvedPath = path.join(projectRoot, importPath);
    return normalizePath(resolvedPath);
  }

  // Handle aliased imports
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

  // For other imports (like external packages), return the original path
  // This will be filtered out later in the FDG builder
  return normalizePath(path.resolve(currentFileDir, importPath));
}

module.exports = {
  resolveAlias,
}; 