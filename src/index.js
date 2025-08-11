const path = require('path');
const { walk } = require('./fileWalker');
const { parseFile } = require('./astParser');
const { outputDir } = require('./config');
const { writeJsonFile, normalizePath } = require('./utils/fileUtils');

const { extractComponents } = require('./extractors/componentExtractor');
const { extractStates } = require('./extractors/stateExtractor');
const { extractProps } = require('./extractors/propExtractor');
const { extractHooks } = require('./extractors/hookExtractor');
const { extractEventHandlers } = require('./extractors/eventHandlerExtractor');
const { extractJsxElements } = require('./extractors/jsxElementExtractor');
const { extractContexts } = require('./extractors/contextExtractor');
const { extractRoutes } = require('./extractors/routeExtractor');

const { buildFdg } = require('./fdgBuilder');
const { buildKg } = require('./kgNodeBuilder');
const { buildUnifiedGraph } = require('./unifiedGraphBuilder');
const { extractCssSelectors } = require('./extractors/cssExtractor');

async function main() {
  console.log('Starting analysis...');
  const files = walk();
  const kgNodesByFile = new Map();

  for (const filePath of files) {
    const normalizedPath = normalizePath(filePath);
    const fileType = path.extname(filePath).substring(1);
    const isStyle = ['css', 'scss', 'sass'].includes(fileType);
    
    const ast = parseFile(filePath);
    const fileNodes = [];

    // For style files, we don't have AST but we still want to include them in the graph
    if (!ast && !isStyle) continue;

    if (ast) {
      const components = extractComponents({ ast, filePath: normalizedPath });
      console.log(`File: ${normalizedPath} - Components found: ${components.length}`);
      if (components.length > 0) {
        components.forEach(comp => console.log(`  - ${comp.name} (${comp.exportType})`));
      }
      fileNodes.push(...components);

      // Extract all related data for each component
      components.forEach((component) => {
        const componentName = component.name;
        
        // Extract states, props, hooks, event handlers, and JSX elements
        const states = extractStates({ ast, componentName, filePath: normalizedPath });
        const props = extractProps({ ast, componentName, filePath: normalizedPath });
        const hooks = extractHooks({ ast, componentName, filePath: normalizedPath });
        const eventHandlers = extractEventHandlers({ ast, componentName, filePath: normalizedPath });
        const jsxElements = extractJsxElements({ ast, componentName, filePath: normalizedPath });
        
        // Add all extracted data to file nodes
        fileNodes.push(...states, ...props, ...hooks, ...eventHandlers, ...jsxElements);
        
        // Update the component node with its related data using IDs
        component.props = props.map(prop => prop.id);
        component.hooksUsed = hooks.map(hook => hook.id);
        // Note: contextUsed will be updated later when we extract contexts
      });

      // Extract contexts and routes
      const contexts = extractContexts({ ast, filePath: normalizedPath });
      const { routes, declareRoutes } = extractRoutes({ ast, filePath: normalizedPath });
      fileNodes.push(...contexts, ...routes);

      // Update component nodes with context information using IDs
      components.forEach((component) => {
        const componentName = component.name;
        const componentContexts = contexts.filter(context => 
          context.componentName === componentName
        );
        component.contextUsed = componentContexts.map(context => context.id);
      });
    } else if (isStyle) {
      // Parse CSS/SCSS/SASS and add CSSSelector nodes
      console.log(`File: ${normalizedPath} - Style file (CSS analysis)`);
      const cssSelectors = extractCssSelectors({ filePath: normalizedPath });
      fileNodes.push(...cssSelectors);
    }

    kgNodesByFile.set(normalizedPath, fileNodes);
  }

  const fdg = buildFdg(files, kgNodesByFile);

  // Build CSS link edges and add Tailwind/InlineStyle nodes before KG aggregation
  const { buildCssLinks } = require('./cssLinkBuilder');
  const cssLinkResult = buildCssLinks(fdg, kgNodesByFile);

  let kg = buildKg(kgNodesByFile);

  // Deduplicate props, hooksUsed, and contextUsed for each component node
  kg.nodes.forEach(node => {
    if (node.type === 'Component') {
      if (Array.isArray(node.props)) {
        node.props = Array.from(new Set(node.props));
      }
      if (Array.isArray(node.hooksUsed)) {
        node.hooksUsed = Array.from(new Set(node.hooksUsed));
      }
      if (Array.isArray(node.contextUsed)) {
        node.contextUsed = Array.from(new Set(node.contextUsed));
      }
    }
  });

  // Update declareRoutes in FDG
  fdg.nodes.forEach((node) => {
    const fileKgNodes = kgNodesByFile.get(node.fileId);
    if (fileKgNodes) {
      const hasRoutes = fileKgNodes.some((kgNode) => kgNode.type === 'Route');
      if (hasRoutes) {
        node.declareRoutes = true;
      }
    }
  });

  writeJsonFile(path.join(outputDir, 'fdg.json'), fdg);

  // Build inter-file KG edges
  const { buildInterFileKgEdges } = require('./interFileKgEdgeBuilder');
  const interFileEdges = buildInterFileKgEdges(fdg, kgNodesByFile);

  // Merge inter-file edges into KG
  if (!Array.isArray(kg.edges)) kg.edges = [];
  // Add CSS edges first
  if (cssLinkResult && Array.isArray(cssLinkResult.edges)) {
    kg.edges.push(...cssLinkResult.edges);
  }
  kg.edges.push(...interFileEdges);

  // Write updated KG (with intra-file and inter-file edges)
  writeJsonFile(path.join(outputDir, 'kg.json'), kg);

  // Build and write unified graph with connecting edges
  const unified = buildUnifiedGraph(fdg, kg, kgNodesByFile);
  writeJsonFile(path.join(outputDir, 'unified.json'), unified);

  console.log('Analysis complete.');
}

main().catch((error) => console.error('Unhandled error:', error)); 