const crypto = require('crypto');

/**
 * Creates a unique ID for a KG node.
 * @param {string} type - The type of the node (e.g., 'Component', 'State').
 * @param {string} name - The name of the node.
 * @param {string} [componentName] - The name of the component the node belongs to.
 * @returns {string} The unique ID.
 */
function createKgNodeId(type, name, componentName) {
  if (componentName) {
    return `${type}:${componentName}:${name}`;
  }
  return `${type}:${name}`;
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