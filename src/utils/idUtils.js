const crypto = require('crypto');
const path = require('path');
const { projectRoot } = require('../config');

/**
 * Creates a unique ID for a KG node with optional component and file context.
 * Formats:
 * - With component and file: `${type}:${componentName}:${name}:${fileBaseName}`
 * - With component only: `${type}:${componentName}:${name}`
 * - With file only: `${type}:${name}:${fileBaseName}`
 * - Otherwise: `${type}:${name}`
 *
 * @param {string} type - The type of the node (e.g., 'Component', 'State').
 * @param {string} name - The name of the node.
 * @param {string} [componentName] - The name of the component the node belongs to.
 * @param {string} [fileBaseName] - The base file name (e.g., `Button.jsx`).
 * @returns {string} The unique ID.
 */
function createKgNodeId(type, name, componentName, fileContext) {
  const rawPath = fileContext || '';
  let relativeFromSrc = '';
  if (rawPath) {
    if (rawPath.includes('/src/')) {
      const idx = rawPath.indexOf('/src/');
      relativeFromSrc = rawPath.substring(idx + '/src/'.length);
    } else {
      try {
        relativeFromSrc = path.relative(projectRoot, rawPath);
      } catch (e) {
        relativeFromSrc = path.basename(rawPath);
      }
    }
  }

  const unique = createHash(`${type}|${name}|${componentName || ''}|${rawPath}`);

  const parts = [type];
  if (componentName) parts.push(componentName);
  if (name) parts.push(name);
  if (relativeFromSrc) parts.push(relativeFromSrc);
  parts.push(unique);
  return parts.join(':');
}

/**
 * Creates a hash of a string.
 * @param {string} input - The string to hash.
 * @returns {string} The hash.
 */
function createHash(input) {
  return crypto.createHash('sha256').update(input).digest('hex').substring(0, 8);
}

module.exports = {
  createKgNodeId,
  createHash,
}; 