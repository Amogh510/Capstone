const traverse = require('@babel/traverse').default;
const { createKgNodeId } = require('../utils/idUtils');
const path = require('path');

/**
 * Extracts context information from an AST.
 * @param {object} ast - The AST to analyze.
 * @param {string} filePath - The path to the file being analyzed.
 * @returns {object[]} An array of context KG nodes.
 */
function extractContexts({ ast, filePath }) {
  const contexts = [];
  let providerFilePath = null;
  const fileBaseName = filePath ? path.basename(filePath) : undefined;

  traverse(ast, {
    CallExpression(path) {
      if (path.node.callee.name === 'createContext') {
        const variableDeclarator = path.findParent((p) =>
          p.isVariableDeclarator()
        );
        if (variableDeclarator) {
          const contextName = variableDeclarator.node.id.name;
          providerFilePath = filePath;
          contexts.push({
            id: createKgNodeId('Context', contextName, undefined, filePath),
            type: 'Context',
            name: contextName,
            usedInComponent: [], // Will be populated later
            providerFilePath,
            providerFileBaseName: fileBaseName,
          });
        }
      } else if (path.node.callee.name === 'useContext') {
        const contextName = path.get('arguments.0').toString();
        // This part is tricky as we need to find which component is using it.
        // This information will be added in a later step.
      }
    },
  });

  return contexts;
}

module.exports = {
  extractContexts,
}; 