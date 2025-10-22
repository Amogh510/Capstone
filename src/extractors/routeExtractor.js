const traverse = require('@babel/traverse').default;
const { createKgNodeId } = require('../utils/idUtils');
const path = require('path');

/**
 * Normalizes and combines route paths, handling both absolute and relative paths.
 * @param {string} parentPath - The parent route path (e.g., "/dashboard")
 * @param {string} childPath - The child route path (e.g., "todos" or "/absolute")
 * @returns {string} The combined path (e.g., "/dashboard/todos")
 */
function combinePaths(parentPath, childPath) {
  // If child is absolute (starts with /), return it as-is
  if (childPath.startsWith('/')) {
    return childPath;
  }
  
  // If parent is empty or child is empty, return the non-empty one
  if (!parentPath) return childPath;
  if (!childPath) return parentPath;
  
  // Combine parent and child, ensuring single slash separator
  const normalizedParent = parentPath.endsWith('/') ? parentPath.slice(0, -1) : parentPath;
  return `${normalizedParent}/${childPath}`;
}

/**
 * Extracts route information from an AST.
 * @param {object} ast - The AST to analyze.
 * @param {string} filePath - The path to the file being analyzed.
 * @returns {{routes: object[], declareRoutes: boolean}} An object containing route KG nodes and a flag.
 */
function extractRoutes({ ast, filePath }) {
  const routes = [];
  let declareRoutes = false;
  const fileBaseName = filePath ? path.basename(filePath) : undefined;
  
  // Stack to track parent route paths for nested routes
  const pathStack = [];

  traverse(ast, {
    JSXElement: {
      enter(nodePath) {
        const openingElement = nodePath.node.openingElement;
        
        if (openingElement.name.name === 'Route') {
          let routePath = '';
          let element = '';

          // Extract path and element attributes
          openingElement.attributes.forEach((attribute) => {
            if (attribute.type === 'JSXAttribute') {
              const attrName = attribute.name.name;

              if (attrName === 'path' && attribute.value) {
                if (attribute.value.type === 'StringLiteral') {
                  routePath = attribute.value.value;
                } else if (attribute.value.type === 'JSXExpressionContainer') {
                  // Handle dynamic paths like path={`/user/${id}`}
                  // For now, we'll skip these or use a placeholder
                  routePath = '<dynamic>';
                }
              } else if (attrName === 'element' && attribute.value) {
                if (attribute.value.type === 'JSXExpressionContainer') {
                  // Try to get a string representation
                  const expr = attribute.value.expression;
                  if (expr.type === 'JSXElement' && expr.openingElement.name.name) {
                    element = expr.openingElement.name.name;
                  } else {
                    element = '<expression>';
                  }
                }
              }
            }
          });

          // Combine with parent path if we have one
          const currentParent = pathStack.length > 0 ? pathStack[pathStack.length - 1] : '';
          const fullPath = routePath ? combinePaths(currentParent, routePath) : currentParent;

          if (fullPath && fullPath !== '<dynamic>') {
            declareRoutes = true;
            routes.push({
              id: createKgNodeId('Route', fullPath, undefined, filePath),
              type: 'Route',
              path: fullPath,
              element,
              filePath,
            });
          }

          // Push current path onto stack for children (if it exists)
          if (fullPath) {
            pathStack.push(fullPath);
          }
        }
      },
      exit(nodePath) {
        const openingElement = nodePath.node.openingElement;
        
        // Pop from stack when leaving a Route element
        if (openingElement.name.name === 'Route' && pathStack.length > 0) {
          pathStack.pop();
        }
      }
    }
  });

  return { routes, declareRoutes };
}

module.exports = {
  extractRoutes,
}; 