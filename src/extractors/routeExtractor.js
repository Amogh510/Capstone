const traverse = require('@babel/traverse').default;
const { createKgNodeId } = require('../utils/idUtils');

/**
 * Extracts route information from an AST.
 * @param {object} ast - The AST to analyze.
 * @param {string} filePath - The path to the file being analyzed.
 * @returns {{routes: object[], declareRoutes: boolean}} An object containing route KG nodes and a flag.
 */
function extractRoutes({ ast, filePath }) {
  const routes = [];
  let declareRoutes = false;

  traverse(ast, {
    JSXOpeningElement(path) {
      if (path.node.name.name === 'Route') {
        let routePath = '';
        let element = '';

        path.get('attributes').forEach((attributePath) => {
          const attribute = attributePath.node;
          const attrName = attribute.name.name;

          if (attrName === 'path') {
            routePath = attribute.value.value;
          } else if (attrName === 'element') {
            element = attributePath.get('value.expression').toString();
            // Remove < and /> from the element string
            element = element.replace(/<|(\s*\/s*)>/g, '');
          }
        });

        if (routePath) {
          declareRoutes = true;
          routes.push({
            id: createKgNodeId('Route', routePath),
            type: 'Route',
            path: routePath,
            element,
            filePath,
          });
        }
      }
    },
  });

  return { routes, declareRoutes };
}

module.exports = {
  extractRoutes,
}; 