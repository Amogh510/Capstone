const traverse = require('@babel/traverse').default;
const { createKgNodeId } = require('../utils/idUtils');
const path = require('path');

/**
 * Extracts hook information from a component's AST.
 * @param {object} ast - The AST of the component file.
 * @param {string} componentName - The name of the component.
 * @returns {object[]} An array of hook KG nodes.
 */
function extractHooks({ ast, componentName, filePath }) {
  const hooks = [];
  const fileBaseName = filePath ? path.basename(filePath) : undefined;
  const reactHooks = new Set(['useEffect', 'useContext', 'useMemo', 'useCallback', 'useRef']);
  const componentId = filePath ? createKgNodeId('Component', componentName, undefined, filePath) : undefined;

  traverse(ast, {
    CallExpression(path) {
      const { callee } = path.node;
      const hookName = callee.name;

      if (hookName && hookName.startsWith('use')) {
        const isCustom = !reactHooks.has(hookName) && hookName !== 'useState' && hookName !== 'useReducer';
        const params = path.get('arguments').map((arg) => arg.toString());

        hooks.push({
          id: createKgNodeId('Hook', hookName, componentName, filePath),
          type: 'Hook',
          hookName,
          isCustom,
          params,
          calledInComponent: componentId || componentName,
          calledInComponentId: componentId,
        });
      }
    },
  });

  return hooks;
}

module.exports = {
  extractHooks,
}; 