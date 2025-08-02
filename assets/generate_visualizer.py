#!/usr/bin/env python3
"""
Generate data visualizer page for WeightWatch project
"""

import json
import html
import base64
import gzip
from pathlib import Path

def load_data(results_dir):
    """Load all JSON data files from results directory, keeping only summaries"""
    data = {}
    results_path = Path(results_dir)
    
    for json_file in results_path.glob("*.json"):
        filename = json_file.stem  # e.g., "10_0"
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                full_data = json.load(f)
                # Compress and encode examples to reduce file size
                min_cleaned = full_data.get('min_cleaned', [])
                max_cleaned = full_data.get('max_cleaned', [])
                
                min_cleaned_b64 = []
                for example in min_cleaned:
                    if isinstance(example, str):
                        compressed = gzip.compress(example.encode('utf-8'))
                        encoded = base64.b64encode(compressed).decode('ascii')
                        min_cleaned_b64.append(encoded)
                
                max_cleaned_b64 = []
                for example in max_cleaned:
                    if isinstance(example, str):
                        compressed = gzip.compress(example.encode('utf-8'))
                        encoded = base64.b64encode(compressed).decode('ascii')
                        max_cleaned_b64.append(encoded)
                
                # Extract min and max values from stats
                min_stats = full_data.get('min_stats', [])
                max_stats = full_data.get('max_stats', [])
                
                min_values = []
                max_values = []
                
                # Extract min values (index 0) from min_stats
                for stat in min_stats:
                    if isinstance(stat, list) and len(stat) > 0:
                        min_values.append(stat[0])
                
                # Extract max values (index 1) from max_stats  
                for stat in max_stats:
                    if isinstance(stat, list) and len(stat) > 1:
                        max_values.append(stat[1])
                
                # Keep the fields we need, with base64 encoded examples
                filtered_data = {
                    'us_desc': full_data.get('us_desc', ''),
                    'min_pattern': full_data.get('min_pattern', ''),
                    'max_pattern': full_data.get('max_pattern', ''),
                    'min_hist': full_data.get('min_hist', []),
                    'max_hist': full_data.get('max_hist', []),
                    'min_cleaned_b64': min_cleaned_b64,
                    'max_cleaned_b64': max_cleaned_b64,
                    'min_values': min_values,
                    'max_values': max_values
                }
                data[filename] = filtered_data
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    print(f'Total directions: {len(data)}')
    
    return data

def generate_html(user_data, assistant_data):
    """Generate the HTML content for the visualizer"""
    
    # Use user_data as default, but we'll load both datasets
    data = user_data
    
    # Extract layers and directions from user data
    layers = set()
    directions = {}
    
    for key, value in data.items():
        us_desc = value.get('us_desc', '')
        if '_u' in us_desc:
            layer_part = us_desc.split('_')[0]
            if layer_part.startswith(('D', 'O')):
                layer_num = int(layer_part[1:])
                layers.add(layer_num)
                if layer_num not in directions:
                    directions[layer_num] = {'D': [], 'O': []}
                
                # Separate D and O directions
                if layer_part.startswith('D'):
                    directions[layer_num]['D'].append(us_desc)
                else:
                    directions[layer_num]['O'].append(us_desc)
    
    layers = sorted(layers)
    
    # Convert data to JSON string with proper escaping for JavaScript
    user_data_json = json.dumps(user_data, ensure_ascii=True, separators=(',', ':'))
    assistant_data_json = json.dumps(assistant_data, ensure_ascii=True, separators=(',', ':'))
    directions_json = json.dumps({str(k): {dk: sorted(list(dv)) for dk, dv in v.items()} for k, v in directions.items()}, ensure_ascii=True, separators=(',', ':'))
    
    # Generate layer buttons
    layer_buttons = '\n'.join(f'                        <div class="radio-button" data-layer="{layer}">Layer {layer}</div>' for layer in layers)
    
    html_content = f'''<!DOCTYPE html>
<html lang="en-GB">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WeightWatch Data Visualizer</title>
    <meta name="description" content="Interactive data visualizer for WeightWatch LLM monitoring results.">
    
    <link rel="stylesheet" type="text/css" media="all" href="assets/stylesheets/main_free.css" />
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/9.18.1/styles/foundation.min.css">
    <link href="assets/fontawesome-free-6.6.0-web/css/all.min.css" rel="stylesheet">
    
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/pako/2.1.0/pako.min.js"></script>
    
    <script type="text/x-mathjax-config">
        MathJax.Hub.Config({{
            "HTML-CSS": {{
              scale: 95,
              fonts: ["Gyre-Pagella"],
              imageFont: null,
              undefinedFamily: "'Arial Unicode MS', cmbright"
            }},
            tex2jax: {{
                inlineMath: [ ['$','$'], ["\\\\(","\\\\)"] ],
                processEscapes: true
              }}
          }});
    </script>
    <script type="text/javascript"
    src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
    </script>
    
    <style>
        .control-panel {{
            background: #f8f9fa;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            border: 1px solid #dee2e6;
        }}
        
        .control-group {{
            margin-bottom: 20px;
        }}
        
        .control-group label {{
            display: block;
            font-weight: bold;
            margin-bottom: 10px;
            font-size: 16px;
        }}
        
        .button-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 10px;
        }}
        
        .radio-button {{
            padding: 6px 12px;
            border: 2px solid #ddd;
            border-radius: 4px;
            background: white;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.2s;
            min-width: 50px;
            text-align: center;
            user-select: none;
        }}
        
        .radio-button:hover {{
            border-color: #007bff;
            background: #f8f9fa;
        }}
        
        .radio-button.active {{
            background: #5fb8e6;
            color: white;
            border-color: #87ceeb;
        }}
        
        .radio-button:disabled {{
            opacity: 0.5;
            cursor: not-allowed;
        }}
        
        .direction-section {{
            margin-bottom: 15px;
        }}
        
        .direction-section h4 {{
            margin: 10px 0 8px 0;
            color: #555;
            font-size: 14px;
        }}
        
        .visualization-container {{
            margin: 30px 0;
        }}
        
        .pattern-container {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin: 20px 0;
        }}
        
        .examples-section {{
            margin: 30px 0;
            width: 100%;
        }}
        
        .examples-section h3 {{
            margin-bottom: 20px;
            color: #333;
            border-bottom: 2px solid #007bff;
            padding-bottom: 10px;
        }}
        
        .pattern-box {{
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 5px;
            background: #f9f9f9;
            cursor: pointer;
            transition: all 0.2s;
        }}
        
        .pattern-box:hover {{
            border-color: #007bff;
            background: #f0f8ff;
            transform: translateY(-2px);
        }}
        
        .pattern-box.clicked {{
            animation: clickPulse 0.3s ease-out;
        }}
        
        @keyframes clickPulse {{
            0% {{ transform: scale(1); }}
            50% {{ transform: scale(1.05); }}
            100% {{ transform: scale(1); }}
        }}
        
        .pattern-box h4 {{
            margin-top: 0;
            color: #333;
        }}
        
        .pattern-text {{
            font-size: 14px;
            line-height: 1.5;
            color: #555;
            white-space: pre-wrap;
        }}
        
        .histogram-container {{
            margin: 20px 0;
        }}
        
        .histogram-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        
        .histogram-box {{
            height: 350px;
        }}
        
        .examples-container {{
            margin: 20px 0;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 8px;
            background: #fff;
            max-height: 500px;
            overflow-y: auto;
            scroll-behavior: smooth;
        }}
        
        .examples-container h4 {{
            margin-top: 0;
            margin-bottom: 20px;
            color: #007bff;
            font-size: 18px;
            border-bottom: 1px solid #eee;
            padding-bottom: 10px;
        }}
        
        .content-warning {{
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 4px;
            padding: 10px;
            margin-bottom: 15px;
            font-size: 13px;
            color: #856404;
        }}
        
        .example-header {{
            margin-bottom: 10px;
            font-size: 14px;
            display: flex;
            gap: 8px;
            align-items: center;
        }}
        
        .example-badge {{
            display: inline-block;
            padding: 4px 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            background: #f8f9fa;
            font-size: 12px;
            font-weight: normal;
            color: #495057;
        }}
        
        .conversation-badge {{
            background: white;
            border-color: white;
            color: black;
            font-family: "Poppins", serif;
            padding-left: 0px !important;
            padding-right: 0px !important;
            font-size: 14px !important;
        }}
        
        .value-badge {{
            background: white;
            border-color: #90caf9;
            color: #1565c0;
            font-family: "Poppins", serif;
        }}
        
        .example-item {{
            margin-bottom: 15px;
            padding: 10px;
            border-left: 3px solid #ddd;
            background: #f8f9fa;
            font-family: 'Courier New', monospace;
            font-size: 13px;
            line-height: 1.4;
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
        
        .example-item:last-child {{
            margin-bottom: 0;
        }}
        
        .search-container {{
            margin: 20px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
            border: 1px solid #dee2e6;
        }}
        
        .search-controls {{
            display: flex;
            gap: 15px;
            align-items: flex-start;
            margin-bottom: 15px;
            flex-wrap: wrap;
        }}
        
        .search-input-group {{
            flex: 1;
            min-width: 300px;
        }}
        
        .search-mode-group {{
            display: flex;
            flex-direction: column;
            gap: 8px;
            flex-shrink: 0;
        }}
        
        .search-mode-group label {{
            font-weight: bold;
            font-size: 14px;
            color: #333;
            margin: 0;
        }}
        
        .search-mode-buttons {{
            display: flex;
            gap: 5px;
        }}
        
        .search-mode-button {{
            padding: 8px 16px !important;
            border: 2px solid #ddd !important;
            border-radius: 4px !important;
            background: white !important;
            cursor: pointer !important;
            font-size: 13px !important;
            transition: all 0.2s !important;
            user-select: none !important;
            font-weight: normal !important;
            color: #333 !important;
            text-align: center !important;
            display: inline-block !important;
            min-width: auto !important;
        }}
        
        .search-mode-button:hover {{
            border-color: #007bff !important;
            background: #f8f9fa !important;
            transform: none !important;
        }}
        
        .search-mode-button.active {{
            background: #007bff !important;
            color: white !important;
            border-color: #0056b3 !important;
        }}
        
        .search-box {{
            width: 90%;
            padding: 12px 15px;
            font-size: 16px;
            border: 2px solid #ddd;
            border-radius: 6px;
            margin-bottom: 10px;
            transition: border-color 0.2s;
        }}
        
        .search-box:focus {{
            outline: none;
            border-color: #007bff;
            box-shadow: 0 0 0 2px rgba(0, 123, 255, 0.25);
        }}
        
        .search-results {{
            font-size: 14px;
            color: #666;
            margin-top: 10px;
        }}
        
        .search-highlight {{
            border-color: #87ceeb !important;
            box-shadow: 0 0 0 1px #87ceeb !important;
        }}
        
        .text-highlight {{
            background-color: #1e3a8a;
            color: white;
            padding: 2px 4px;
            border-radius: 3px;
        }}
        
        .loading {{
            text-align: center;
            padding: 40px;
            color: #666;
        }}
        
        .data-switch-container {{
            background: #f0f0f0;
            padding: 15px 0;
            border-bottom: 1px solid #ddd;
        }}
        
        .data-switch-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 20px;
            flex-wrap: wrap;
            gap: 20px;
        }}
        
        .data-switch-group {{
            display: flex;
            align-items: center;
            gap: 15px;
        }}
        
        .quick-nav-group {{
            display: flex;
            align-items: center;
            gap: 15px;
        }}
        
        .quick-nav-group label {{
            font-weight: bold;
            font-size: 16px;
            color: #333;
            margin: 0;
            white-space: nowrap;
        }}
        
        .quick-nav-buttons {{
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }}
        
        .data-switch-group label {{
            font-weight: bold;
            font-size: 16px;
            color: #333;
            margin: 0;
        }}
        
        .data-switch-buttons {{
            display: flex;
            gap: 5px;
        }}
        
        .data-switch-button {{
            padding: 10px 20px !important;
            border: 2px solid #ddd !important;
            border-radius: 6px !important;
            background: white !important;
            cursor: pointer !important;
            font-size: 14px !important;
            font-weight: bold !important;
            transition: all 0.2s !important;
            user-select: none !important;
            color: #333 !important;
            text-align: center !important;
            display: inline-block !important;
            min-width: 80px !important;
        }}
        
        .data-switch-button:hover {{
            border-color: #007bff !important;
            background: #f8f9fa !important;
            transform: none !important;
        }}
        
        .data-switch-button.active {{
            background: #007bff !important;
            color: white !important;
            border-color: #0056b3 !important;
        }}
        
        .quick-nav-button {{
            padding: 6px 12px !important;
            border: 1px solid #ddd !important;
            border-radius: 4px !important;
            background: white !important;
            cursor: pointer !important;
            font-size: 14px !important;
            font-weight: normal !important;
            transition: all 0.2s !important;
            user-select: none !important;
            color: #333 !important;
            text-align: center !important;
            display: inline-block !important;
            min-width: 50px !important;
        }}
        
        .quick-nav-button:hover {{
            border-color: #28a745 !important;
            background: #f8f9fa !important;
            transform: none !important;
        }}
        
    </style>
</head>

<body>
    <!-- Title Page -->
    <div class="container blog" id="first-content" style="background-color: #E0E4E6;">
        <div class="blog-title no-cover">
            <div class="blog-intro">
                <div style="text-align: center;">
                    <h1 class="title">WeightWatch Visualizer for Qwen-2.5 7B</h1>
                    <p class="abstract">
                        This visualizer allows you to explore the activation patterns discovered by WeightWatch across different layers and SVD directions. For each singular direction, 10 minimally and maximally activated samples are shown (with rank #1, #4 ... #31), as well as interpretations from o4-mini. Results may be slightly different from the paper since we reran sample collection and truncated samples at maximally firing tokens.
                    </p>

                    <!-- Using FontAwesome Free -->
                    <div class="info">
                        <div>
                            <a href="index.html" class="button icon" style="background-color: rgba(255, 255, 255, 0.2)">Main Page <i class="fa-solid fa-home"></i></a>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Data Source Switch -->
        <div class="data-switch-container">
            <div class="data-switch-row">
                <div class="data-switch-group">
                    <label>Role:</label>
                    <div class="data-switch-buttons">
                        <div class="data-switch-button active" data-source="user">User</div>
                        <div class="data-switch-button" data-source="assistant">Assistant</div>
                    </div>
                </div>
                <div class="quick-nav-group">
                    <label>You may start from these:</label>
                    <div class="quick-nav-buttons" id="quick-nav-buttons">
                        <!-- Buttons will be populated by JavaScript based on current data source -->
                    </div>
                </div>
            </div>
        </div>
    </div>

    <div class="container">
        <main>
            <div class="wrapper">
                <div class="search-container">
                    <div class="search-controls">
                        <div class="search-input-group">
                            <input type="text" id="pattern-search" class="search-box" placeholder="Search...">
                        </div>
                        <div class="search-mode-group">
                            <label>Search Mode:</label>
                            <div class="search-mode-buttons">
                                <div class="search-mode-button active" data-mode="pattern">Pattern</div>
                                <div class="search-mode-button" data-mode="transcript">Transcript</div>
                                <div class="search-mode-button" data-mode="extremes">Extremes</div>
                            </div>
                        </div>
                    </div>
                    <div id="search-results" class="search-results">Search patterns, transcripts, or extreme examples (e.g., 'russian', 'translate', 'code')...</div>
                </div>
                
                <div class="control-panel">
                    <div class="control-group">
                        <label>Layer:</label>
                        <div class="button-row" id="layer-buttons">
{layer_buttons}
                        </div>
                    </div>
                    
                    <div class="control-group">
                        <label>Direction:</label>
                        <div id="direction-controls">
                            <div class="direction-section">
                                <h4>$\\Delta O_{{\\text{{proj}}}}$ directions (U0-U19):</h4>
                                <div class="button-row" id="o-direction-buttons"></div>
                            </div>
                            <div class="direction-section">
                                <h4>$\\Delta W_{{\\text{{down}}}}$ projections (U0-U19):</h4>
                                <div class="button-row" id="d-direction-buttons"></div>
                            </div>
                        </div>
                    </div>
                </div>

                <div id="content-area">
                    <div class="loading">
                        <p>Select a layer and direction to view the data visualization.</p>
                    </div>
                </div>
            </div>
        </main>
    </div>

    <!-- Footer Page -->
    <footer style="font-style: normal !important;">
        <div class="container">
            <p style="text-align: center;">
                Built with ❤️ and Claude Code. If you love playing with this, also check out <a href="https://www.neuronpedia.org/">Neuronpedia</a> on SAEs.
            </p>
        </div>    
    </footer>

    <script>
        // Data storage
        const userDataRaw = {user_data_json};
        const assistantDataRaw = {assistant_data_json};
        
        // Decode all base64 examples upfront for faster searching
        const userData = decodeAllExamples(userDataRaw);
        const assistantData = decodeAllExamples(assistantDataRaw);
        
        let data = userData; // Start with user data
        let currentDataSource = 'user';
        
        // DOM elements
        const contentArea = document.getElementById('content-area');
        
        // Direction mappings for each layer
        const directions = {directions_json};
        
        // Current selection
        let currentLayer = null;
        let currentDirection = null;
        let currentSearchMode = 'pattern';
        
        // Initialize
        document.addEventListener('DOMContentLoaded', function() {{
            setupDataSwitch();
            setupQuickNav();
            setupLayerButtons();
            updateDirectionButtons();
            setupSearch();
            setupSearchModes();
            
            // Render MathJax after content is loaded
            if (typeof MathJax !== 'undefined') {{
                MathJax.Hub.Queue(["Typeset", MathJax.Hub]);
            }}
        }});
        
        function getDirectionType(direction) {{
            if (!direction) return null;
            return direction.startsWith('O') ? 'O' : 'D';
        }}
        
        function getDirectionUNumber(direction) {{
            if (!direction) return null;
            const match = direction.match(/_u(\\d+)$/);
            return match ? parseInt(match[1]) : null;
        }}
        
        function findMatchingDirection(oldDirection, newLayer) {{
            if (!oldDirection || !newLayer || !directions[newLayer]) return null;
            
            const dirType = getDirectionType(oldDirection);
            const uNum = getDirectionUNumber(oldDirection);
            
            if (dirType === null || uNum === null) return null;
            
            const layerDirections = directions[newLayer][dirType] || [];
            return layerDirections.find(dir => getDirectionUNumber(dir) === uNum) || null;
        }}
        
        function setupLayerButtons() {{
            const layerButtons = document.querySelectorAll('#layer-buttons .radio-button');
            layerButtons.forEach(button => {{
                button.addEventListener('click', function() {{
                    // Remove active class from all layer buttons
                    layerButtons.forEach(b => b.classList.remove('active'));
                    // Add active class to clicked button
                    this.classList.add('active');
                    // Update current layer
                    currentLayer = this.getAttribute('data-layer');
                    
                    // Try to find matching direction in new layer (same U number and type)
                    const matchingDirection = findMatchingDirection(currentDirection, currentLayer);
                    
                    if (matchingDirection) {{
                        // Found matching direction, update to new layer's version
                        currentDirection = matchingDirection;
                    }} else {{
                        // No matching direction found, clear selection
                        currentDirection = null;
                        clearContent();
                    }}
                    
                    // Update direction buttons (this will maintain active state if we found a match)
                    updateDirectionButtons();
                    
                    // If we have a direction, reload visualization
                    if (currentDirection) {{
                        loadVisualization(currentDirection);
                    }}
                }});
            }});
        }}
        
        function updateDirectionButtons() {{
            const oButtons = document.getElementById('o-direction-buttons');
            const dButtons = document.getElementById('d-direction-buttons');
            
            // Store current search query to re-apply highlights
            const searchQuery = document.getElementById('pattern-search').value.toLowerCase().trim();
            
            // Clear existing buttons
            oButtons.innerHTML = '';
            dButtons.innerHTML = '';
            
            if (!currentLayer || !directions[currentLayer]) {{
                return;
            }}
            
            // Create O direction buttons (U0-U19)
            const oDirections = directions[currentLayer]['O'] || [];
            oDirections.sort((a, b) => {{
                const aNum = parseInt(a.split('_u')[1]);
                const bNum = parseInt(b.split('_u')[1]);
                return aNum - bNum;
            }});
            
            oDirections.forEach(direction => {{
                const uNum = direction.split('_u')[1];
                const button = document.createElement('div');
                button.className = 'radio-button';
                if (direction === currentDirection) {{
                    button.classList.add('active');
                }}
                button.setAttribute('data-direction', direction);
                button.textContent = `U${{uNum}}`;
                button.addEventListener('click', function() {{
                    selectDirection(this, direction);
                }});
                oButtons.appendChild(button);
            }});
            
            // Create D direction buttons (U0-U19)  
            const dDirections = directions[currentLayer]['D'] || [];
            dDirections.sort((a, b) => {{
                const aNum = parseInt(a.split('_u')[1]);
                const bNum = parseInt(b.split('_u')[1]);
                return aNum - bNum;
            }});
            
            dDirections.forEach(direction => {{
                const uNum = direction.split('_u')[1];
                const button = document.createElement('div');
                button.className = 'radio-button';
                if (direction === currentDirection) {{
                    button.classList.add('active');
                }}
                button.setAttribute('data-direction', direction);
                button.textContent = `U${{uNum}}`;
                button.addEventListener('click', function() {{
                    selectDirection(this, direction);
                }});
                dButtons.appendChild(button);
            }});
            
            // Re-apply search highlights if there's an active search
            if (searchQuery) {{
                const resultsDiv = document.getElementById('search-results');
                performSearch(searchQuery, resultsDiv);
            }}
        }}
        
        function selectDirection(buttonElement, direction) {{
            // Remove active class from all direction buttons
            document.querySelectorAll('#o-direction-buttons .radio-button, #d-direction-buttons .radio-button')
                .forEach(b => b.classList.remove('active'));
            // Add active class to clicked button
            buttonElement.classList.add('active');
            // Update current direction
            currentDirection = direction;
            loadVisualization(direction);
        }}
        
        function clearContent() {{
            contentArea.innerHTML = '<div class="loading"><p>Select a layer and direction to view the data visualization.</p></div>';
        }}
        
        function loadVisualization(direction) {{
            // Find the data entry for this direction
            let dataEntry = null;
            let dataKey = null;
            
            for (const [key, value] of Object.entries(data)) {{
                if (value.us_desc === direction) {{
                    dataEntry = value;
                    dataKey = key;
                    break;
                }}
            }}
            
            if (!dataEntry) {{
                contentArea.innerHTML = '<div class="loading"><p>No data found for this direction.</p></div>';
                return;
            }}
            
            // Create container div
            const container = document.createElement('div');
            container.className = 'visualization-container';
            
            // Create title
            const title = document.createElement('h3');
            title.textContent = `Direction: ${{direction}} (File: ${{dataKey}}.json)`;
            container.appendChild(title);
            
            // Create pattern container FIRST
            const patternContainer = document.createElement('div');
            patternContainer.className = 'pattern-container';
            
            // Min pattern box
            const minBox = document.createElement('div');
            minBox.className = 'pattern-box';
            minBox.setAttribute('data-type', 'min');
            const minTitle = document.createElement('h4');
            minTitle.textContent = 'Minimum Activation Pattern (click to jump to examples)';
            const minText = document.createElement('div');
            minText.className = 'pattern-text';
            minText.textContent = dataEntry.min_pattern || 'No pattern available';
            minBox.appendChild(minTitle);
            minBox.appendChild(minText);
            
            // Max pattern box
            const maxBox = document.createElement('div');
            maxBox.className = 'pattern-box';
            maxBox.setAttribute('data-type', 'max');
            const maxTitle = document.createElement('h4');
            maxTitle.textContent = 'Maximum Activation Pattern (click to jump to examples)';
            const maxText = document.createElement('div');
            maxText.className = 'pattern-text';
            maxText.textContent = dataEntry.max_pattern || 'No pattern available';
            maxBox.appendChild(maxTitle);
            maxBox.appendChild(maxText);
            
            // Add click handlers for pattern boxes - scroll to examples
            minBox.addEventListener('click', function() {{
                this.classList.add('clicked');
                setTimeout(() => this.classList.remove('clicked'), 300);
                scrollToExamples('min-examples');
            }});
            
            maxBox.addEventListener('click', function() {{
                this.classList.add('clicked');
                setTimeout(() => this.classList.remove('clicked'), 300);
                scrollToExamples('max-examples');
            }});
            
            patternContainer.appendChild(minBox);
            patternContainer.appendChild(maxBox);
            container.appendChild(patternContainer);
            
            // Create histogram container AFTER patterns
            const histContainer = document.createElement('div');
            histContainer.className = 'histogram-container';
            const histGrid = document.createElement('div');
            histGrid.className = 'histogram-grid';
            
            const minHistDiv = document.createElement('div');
            minHistDiv.className = 'histogram-box';
            minHistDiv.id = 'min-histogram-plot';
            
            const maxHistDiv = document.createElement('div');
            maxHistDiv.className = 'histogram-box';
            maxHistDiv.id = 'max-histogram-plot';
            
            histGrid.appendChild(minHistDiv);
            histGrid.appendChild(maxHistDiv);
            histContainer.appendChild(histGrid);
            container.appendChild(histContainer);
            
            // Create examples section
            const examplesSection = document.createElement('div');
            examplesSection.className = 'examples-section';
            
            // Min examples
            const minExamples = dataEntry.min_cleaned || [];
            const minValues = dataEntry.min_values || [];
            const minExamplesContainer = createExamplesContainer('min-examples', 'Minimum Activation Examples', minExamples, minValues);
            examplesSection.appendChild(minExamplesContainer);
            
            // Max examples  
            const maxExamples = dataEntry.max_cleaned || [];
            const maxValues = dataEntry.max_values || [];
            const maxExamplesContainer = createExamplesContainer('max-examples', 'Maximum Activation Examples', maxExamples, maxValues);
            examplesSection.appendChild(maxExamplesContainer);
            
            container.appendChild(examplesSection);
            
            // Replace content
            contentArea.innerHTML = '';
            contentArea.appendChild(container);
            
            // Create histogram plots
            createHistograms(dataEntry);
            
            // Re-apply search highlighting if there's an active search
            const searchQuery = document.getElementById('pattern-search').value.toLowerCase().trim();
            if (searchQuery) {{
                // Always highlight everywhere regardless of search mode
                highlightCurrentPatternText(searchQuery);
                highlightCurrentTranscriptText(searchQuery);
            }}
        }}
        
        function createHistograms(dataEntry) {{
            const minHist = dataEntry.min_hist;
            const maxHist = dataEntry.max_hist;
            
            if (!minHist || !minHist[0] || !minHist[1] || !maxHist || !maxHist[0] || !maxHist[1]) {{
                document.getElementById('min-histogram-plot').innerHTML = '<p>No histogram data available</p>';
                document.getElementById('max-histogram-plot').innerHTML = '<p>No histogram data available</p>';
                return;
            }}
            
            // Create bin centers from bin edges
            const minBinCenters = [];
            const maxBinCenters = [];
            
            for (let i = 0; i < minHist[1].length - 1; i++) {{
                minBinCenters.push((minHist[1][i] + minHist[1][i + 1]) / 2);
            }}
            
            for (let i = 0; i < maxHist[1].length - 1; i++) {{
                maxBinCenters.push((maxHist[1][i] + maxHist[1][i + 1]) / 2);
            }}
            
            // Min activation histogram
            const minTrace = {{
                x: minBinCenters,
                y: minHist[0],
                type: 'bar',
                name: 'Min Activation',
                marker: {{
                    color: 'rgba(55, 128, 191, 0.7)',
                    line: {{
                        color: 'rgba(55, 128, 191, 1.0)',
                        width: 1
                    }}
                }}
            }};
            
            const minLayout = {{
                title: 'Minimum Activation Histogram',
                xaxis: {{
                    title: 'Activation Value'
                }},
                yaxis: {{
                    title: 'Frequency'
                }},
                height: 350,
                margin: {{ t: 50, b: 50, l: 50, r: 20 }}
            }};
            
            // Max activation histogram  
            const maxTrace = {{
                x: maxBinCenters,
                y: maxHist[0],
                type: 'bar',
                name: 'Max Activation',
                marker: {{
                    color: 'rgba(219, 64, 82, 0.7)',
                    line: {{
                        color: 'rgba(219, 64, 82, 1.0)',
                        width: 1
                    }}
                }}
            }};
            
            const maxLayout = {{
                title: 'Maximum Activation Histogram',
                xaxis: {{
                    title: 'Activation Value'
                }},
                yaxis: {{
                    title: 'Frequency'
                }},
                height: 350,
                margin: {{ t: 50, b: 50, l: 50, r: 20 }}
            }};
            
            Plotly.newPlot('min-histogram-plot', [minTrace], minLayout);
            Plotly.newPlot('max-histogram-plot', [maxTrace], maxLayout);
        }}
        
        function decodeBase64Examples(base64Array) {{
            const decodedExamples = [];
            for (const b64String of base64Array) {{
                try {{
                    // Decode base64 to get compressed data
                    const decoded = atob(b64String);
                    
                    // Convert string to Uint8Array for decompression
                    const compressedData = new Uint8Array(decoded.length);
                    for (let i = 0; i < decoded.length; i++) {{
                        compressedData[i] = decoded.charCodeAt(i);
                    }}
                    
                    // Decompress using pako (gzip)
                    const decompressed = pako.inflate(compressedData, {{ to: 'string' }});
                    decodedExamples.push(decompressed);
                }} catch (e) {{
                    console.error('Failed to decode/decompress example:', e);
                    decodedExamples.push('[Decoding error]');
                }}
            }}
            return decodedExamples;
        }}
        
        function decodeAllExamples(dataObj) {{
            const decodedData = {{}};
            for (const [key, value] of Object.entries(dataObj)) {{
                decodedData[key] = {{
                    ...value,
                    min_cleaned: decodeBase64Examples(value.min_cleaned_b64 || []),
                    max_cleaned: decodeBase64Examples(value.max_cleaned_b64 || [])
                }};
            }}
            return decodedData;
        }}
        
        function scrollToExamples(elementId) {{
            const element = document.getElementById(elementId);
            if (element) {{
                element.scrollIntoView({{ 
                    behavior: 'smooth', 
                    block: 'start',
                    inline: 'nearest'
                }});
            }}
        }}
        
        function createExamplesContainer(id, title, examples, values) {{
            const container = document.createElement('div');
            container.className = 'examples-container';
            container.id = id;
            
            const titleElement = document.createElement('h4');
            titleElement.textContent = title;
            container.appendChild(titleElement);
            
            // Add content warning
            const warning = document.createElement('div');
            warning.className = 'content-warning';
            warning.innerHTML = '<strong>⚠️ Content Warning:</strong> Examples from WildChat may contain potentially harmful, offensive, or inappropriate content.';
            container.appendChild(warning);
            
            if (examples.length === 0) {{
                const noExamples = document.createElement('div');
                noExamples.className = 'example-item';
                noExamples.textContent = 'No examples available';
                container.appendChild(noExamples);
            }} else {{
                // Show first 15 examples for full-width display
                examples.slice(0, 15).forEach((example, index) => {{
                    const headerDiv = document.createElement('div');
                    headerDiv.className = 'example-header';
                    
                    // Create conversation badge
                    const conversationBadge = document.createElement('span');
                    conversationBadge.className = 'example-badge conversation-badge';
                    conversationBadge.textContent = `Conversation ${{index + 1}}`;
                    headerDiv.appendChild(conversationBadge);
                    
                    // Create value badge if available
                    const value = values && values[index] !== undefined ? values[index] : null;
                    if (value !== null) {{
                        const valueBadge = document.createElement('span');
                        valueBadge.className = 'example-badge value-badge';
                        valueBadge.textContent = value.toFixed(4);
                        headerDiv.appendChild(valueBadge);
                    }}
                    
                    const exampleDiv = document.createElement('div');
                    exampleDiv.className = 'example-item';
                    exampleDiv.textContent = example;
                    
                    container.appendChild(headerDiv);
                    container.appendChild(exampleDiv);
                }});
                
                if (examples.length > 15) {{
                    const moreInfo = document.createElement('div');
                    moreInfo.className = 'example-item';
                    moreInfo.style.fontStyle = 'italic';
                    moreInfo.style.color = '#666';
                    moreInfo.textContent = '... and ' + (examples.length - 15) + ' more examples (showing first 15)';
                    container.appendChild(moreInfo);
                }}
            }}
            
            return container;
        }}
        
        function setupDataSwitch() {{
            const switchButtons = document.querySelectorAll('.data-switch-button');
            
            switchButtons.forEach(button => {{
                button.addEventListener('click', function() {{
                    const newSource = this.getAttribute('data-source');
                    if (newSource === currentDataSource) return; // No change needed
                    
                    // Remove active class from all switch buttons
                    switchButtons.forEach(b => b.classList.remove('active'));
                    // Add active class to clicked button
                    this.classList.add('active');
                    
                    // Switch data source
                    currentDataSource = newSource;
                    data = (newSource === 'user') ? userData : assistantData;
                    
                    // Update quick nav buttons for new data source
                    updateQuickNavButtons();
                    
                    // Preserve current selections but update available options
                    const oldLayer = currentLayer;
                    const oldDirection = currentDirection;
                    const oldSearchQuery = document.getElementById('pattern-search').value;
                    
                    // Update direction buttons for new data
                    updateDirectionButtons();
                    
                    // Try to restore selections if they exist in new data
                    if (oldDirection && data[findDataKeyForDirection(oldDirection)]) {{
                        // Direction exists in new data, reload visualization
                        loadVisualization(oldDirection);
                    }} else {{
                        // Direction doesn't exist, clear content but keep layer/direction selections
                        clearContent();
                    }}
                    
                    // Re-run search if there was one
                    if (oldSearchQuery) {{
                        const resultsDiv = document.getElementById('search-results');
                        performSearch(oldSearchQuery.toLowerCase().trim(), resultsDiv);
                    }}
                }});
            }});
        }}
        
        function setupQuickNav() {{
            updateQuickNavButtons();
        }}
        
        function updateQuickNavButtons() {{
            const quickNavContainer = document.getElementById('quick-nav-buttons');
            quickNavContainer.innerHTML = '';
            
            const userDirections = ['O1_u13', 'O4_u13', 'O14_u12', 'O16_u8'];
            const assistantDirections = ['O8_u13', 'D17_u6', 'D18_u8', 'O18_u15'];
            
            const directionsToShow = currentDataSource === 'user' ? userDirections : assistantDirections;
            
            // Add specific direction buttons
            directionsToShow.forEach(direction => {{
                const button = document.createElement('div');
                button.className = 'quick-nav-button';
                button.textContent = direction;
                button.addEventListener('click', function() {{
                    navigateToDirection(direction);
                }});
                quickNavContainer.appendChild(button);
            }});
            
            // Add random button
            const randomButton = document.createElement('div');
            randomButton.className = 'quick-nav-button';
            randomButton.textContent = 'Random';
            randomButton.addEventListener('click', function() {{
                navigateToRandomDirection();
            }});
            quickNavContainer.appendChild(randomButton);
        }}
        
        function navigateToDirection(direction) {{
            // Parse layer and direction from the direction string (e.g., "O8_u13")
            const match = direction.match(/^([DO])(\\d+)_u(\\d+)$/);
            if (!match) return;
            
            const directionType = match[1];
            const layerNum = match[2];
            const uNum = match[3];
            
            // Find if this direction exists in current data
            let found = false;
            for (const [key, value] of Object.entries(data)) {{
                if (value.us_desc === direction) {{
                    found = true;
                    break;
                }}
            }}
            
            if (found) {{
                // Set layer
                currentLayer = layerNum;
                document.querySelectorAll('#layer-buttons .radio-button').forEach(b => b.classList.remove('active'));
                const layerButton = document.querySelector(`#layer-buttons .radio-button[data-layer="${{layerNum}}"]`);
                if (layerButton) {{
                    layerButton.classList.add('active');
                }}
                
                // Update direction buttons for this layer
                updateDirectionButtons();
                
                // Set direction
                currentDirection = direction;
                document.querySelectorAll('#o-direction-buttons .radio-button, #d-direction-buttons .radio-button').forEach(b => b.classList.remove('active'));
                const directionButton = document.querySelector(`[data-direction="${{direction}}"]`);
                if (directionButton) {{
                    directionButton.classList.add('active');
                }}
                
                // Load visualization
                loadVisualization(direction);
            }} else {{
                alert(`Direction ${{direction}} not found in ${{currentDataSource}} data`);
            }}
        }}
        
        function navigateToRandomDirection() {{
            const dataKeys = Object.keys(data);
            if (dataKeys.length === 0) return;
            
            const randomKey = dataKeys[Math.floor(Math.random() * dataKeys.length)];
            const randomDirection = data[randomKey].us_desc;
            
            navigateToDirection(randomDirection);
        }}
        
        function findDataKeyForDirection(direction) {{
            for (const [key, value] of Object.entries(data)) {{
                if (value.us_desc === direction) {{
                    return key;
                }}
            }}
            return null;
        }}
        
        function setupSearchModes() {{
            const modeButtons = document.querySelectorAll('.search-mode-button');
            
            modeButtons.forEach(button => {{
                button.addEventListener('click', function() {{
                    // Remove active class from all mode buttons
                    modeButtons.forEach(b => b.classList.remove('active'));
                    // Add active class to clicked button
                    this.classList.add('active');
                    // Update current search mode
                    currentSearchMode = this.getAttribute('data-mode');
                    
                    // Update placeholder and help text
                    const searchBox = document.getElementById('pattern-search');
                    const resultsDiv = document.getElementById('search-results');
                    
                    if (currentSearchMode === 'pattern') {{
                        if (searchBox.value === '') {{
                            resultsDiv.textContent = "Search within patterns...";
                        }}
                    }} else if (currentSearchMode === 'transcript') {{
                        if (searchBox.value === '') {{
                            resultsDiv.textContent = "Search within conversation transcripts...";
                        }}
                    }} else {{
                        if (searchBox.value === '') {{
                            resultsDiv.textContent = "Search within most extreme examples (top 1 from each category)...";
                        }}
                    }}
                    
                    // Re-run search if there's a query
                    const query = searchBox.value.toLowerCase().trim();
                    if (query) {{
                        performSearch(query, resultsDiv);
                    }}
                }});
            }});
        }}
        
        function setupSearch() {{
            const searchBox = document.getElementById('pattern-search');
            const resultsDiv = document.getElementById('search-results');
            
            searchBox.addEventListener('input', function() {{
                const query = this.value.toLowerCase().trim();
                
                if (query === '') {{
                    clearSearchHighlights();
                    if (currentSearchMode === 'pattern') {{
                        resultsDiv.textContent = "Search within patterns (e.g., 'russian', 'translate', 'code')...";
                    }} else if (currentSearchMode === 'transcript') {{
                        resultsDiv.textContent = "Search within conversation transcripts...";
                    }} else {{
                        resultsDiv.textContent = "Search within most extreme examples (top 1 from each category)...";
                    }}
                    return;
                }}
                
                performSearch(query, resultsDiv);
            }});
        }}
        
        function performSearch(query, resultsDiv) {{
            clearSearchHighlights();
            
            const matches = {{
                directions: [],
                layersWithMatches: new Set()
            }};
            
            if (currentSearchMode === 'pattern') {{
                // Search through patterns
                for (const [key, value] of Object.entries(data)) {{
                    const minPattern = (value.min_pattern || '').toLowerCase();
                    const maxPattern = (value.max_pattern || '').toLowerCase();
                    
                    if (minPattern.includes(query) || maxPattern.includes(query)) {{
                        matches.directions.push(value.us_desc);
                        
                        // Extract layer number from direction (e.g., "O5_u12" -> 5)
                        const layerMatch = value.us_desc.match(/^[DO](\\d+)_/);
                        if (layerMatch) {{
                            matches.layersWithMatches.add(layerMatch[1]);
                        }}
                    }}
                }}
                
                // Highlight text everywhere (patterns and transcripts)
                highlightCurrentPatternText(query);
                highlightCurrentTranscriptText(query);
                
                // Display results
                const directionCount = matches.directions.length;
                const layerCount = matches.layersWithMatches.size;
                const totalDirections = Object.keys(data).length;
                const percentage = totalDirections > 0 ? (directionCount / totalDirections * 100).toFixed(1) : 0;
                
                if (directionCount === 0) {{
                    resultsDiv.textContent = `No matches found for "${{query}}" in patterns`;
                }} else {{
                    resultsDiv.textContent = `Found ${{directionCount}} direction${{directionCount !== 1 ? 's' : ''}} across ${{layerCount}} layer${{layerCount !== 1 ? 's' : ''}} with patterns matching "${{query}}" (${{percentage}}%)`;
                }}
            }} else if (currentSearchMode === 'transcript') {{
                // Search through transcripts
                for (const [key, value] of Object.entries(data)) {{
                    const minExamples = value.min_cleaned || [];
                    const maxExamples = value.max_cleaned || [];
                    
                    let hasMatch = false;
                    for (const example of [...minExamples, ...maxExamples]) {{
                        if (example.toLowerCase().includes(query)) {{
                            hasMatch = true;
                            break;
                        }}
                    }}
                    
                    if (hasMatch) {{
                        matches.directions.push(value.us_desc);
                        
                        // Extract layer number from direction (e.g., "O5_u12" -> 5)
                        const layerMatch = value.us_desc.match(/^[DO](\\d+)_/);
                        if (layerMatch) {{
                            matches.layersWithMatches.add(layerMatch[1]);
                        }}
                    }}
                }}
                
                // Highlight text everywhere (patterns and transcripts)
                highlightCurrentPatternText(query);
                highlightCurrentTranscriptText(query);
                
                // Display results
                const directionCount = matches.directions.length;
                const layerCount = matches.layersWithMatches.size;
                const totalDirections = Object.keys(data).length;
                const percentage = totalDirections > 0 ? (directionCount / totalDirections * 100).toFixed(1) : 0;
                
                if (directionCount === 0) {{
                    resultsDiv.textContent = `No matches found for "${{query}}" in transcripts`;
                }} else {{
                    resultsDiv.textContent = `Found ${{directionCount}} direction${{directionCount !== 1 ? 's' : ''}} across ${{layerCount}} layer${{layerCount !== 1 ? 's' : ''}} with transcripts containing "${{query}}" (${{percentage}}%)`;
                }}
            }} else {{
                // Search through extremes (top 1 example from each category)
                for (const [key, value] of Object.entries(data)) {{
                    const minExamples = (value.min_cleaned || []).slice(0, 1);
                    const maxExamples = (value.max_cleaned || []).slice(0, 1);
                    
                    let hasMatch = false;
                    for (const example of [...minExamples, ...maxExamples]) {{
                        if (example.toLowerCase().includes(query)) {{
                            hasMatch = true;
                            break;
                        }}
                    }}
                    
                    if (hasMatch) {{
                        matches.directions.push(value.us_desc);
                        
                        // Extract layer number from direction (e.g., "O5_u12" -> 5)
                        const layerMatch = value.us_desc.match(/^[DO](\\d+)_/);
                        if (layerMatch) {{
                            matches.layersWithMatches.add(layerMatch[1]);
                        }}
                    }}
                }}
                
                // Highlight text everywhere (patterns and transcripts)
                highlightCurrentPatternText(query);
                highlightCurrentTranscriptText(query);
                
                // Display results
                const directionCount = matches.directions.length;
                const layerCount = matches.layersWithMatches.size;
                const totalDirections = Object.keys(data).length;
                const percentage = totalDirections > 0 ? (directionCount / totalDirections * 100).toFixed(1) : 0;
                
                if (directionCount === 0) {{
                    resultsDiv.textContent = `No matches found for "${{query}}" in top examples`;
                }} else {{
                    resultsDiv.textContent = `Found ${{directionCount}} direction${{directionCount !== 1 ? 's' : ''}} across ${{layerCount}} layer${{layerCount !== 1 ? 's' : ''}} with top examples containing "${{query}}" (${{percentage}}%)`;
                }}
            }}
            
            // Highlight matching elements
            highlightMatches(matches);
        }}
        
        function highlightMatches(matches) {{
            // Highlight matching layer buttons
            document.querySelectorAll('#layer-buttons .radio-button').forEach(button => {{
                const layer = button.getAttribute('data-layer');
                if (matches.layersWithMatches.has(layer)) {{
                    button.classList.add('search-highlight');
                }}
            }});
            
            // Highlight matching direction buttons
            document.querySelectorAll('#o-direction-buttons .radio-button, #d-direction-buttons .radio-button').forEach(button => {{
                const direction = button.getAttribute('data-direction');
                if (matches.directions.includes(direction)) {{
                    button.classList.add('search-highlight');
                }}
            }});
        }}
        
        function clearSearchHighlights() {{
            document.querySelectorAll('.search-highlight').forEach(element => {{
                element.classList.remove('search-highlight');
            }});
            // Also clear text highlights in patterns
            clearTextHighlights();
        }}
        
        function clearTextHighlights() {{
            document.querySelectorAll('.pattern-text, .example-item').forEach(element => {{
                // Restore original text without highlights
                const originalText = element.textContent;
                element.innerHTML = '';
                element.textContent = originalText;
            }});
        }}
        
        function highlightTextInElement(element, query) {{
            const text = element.textContent;
            const regex = new RegExp(`(${{query.replace(/[.*+?^${{}}()|[\\]\\\\]/g, '\\\\$&')}})`, 'gi');
            const highlightedText = text.replace(regex, '<span class="text-highlight">$1</span>');
            
            if (highlightedText !== text) {{
                element.innerHTML = highlightedText;
                return true;
            }}
            return false;
        }}
        
        function highlightCurrentPatternText(query) {{
            // Highlight text in currently visible pattern boxes
            document.querySelectorAll('.pattern-text').forEach(element => {{
                highlightTextInElement(element, query);
            }});
        }}
        
        function highlightCurrentTranscriptText(query) {{
            // Highlight text in currently visible transcript examples
            document.querySelectorAll('.example-item').forEach(element => {{
                highlightTextInElement(element, query);
            }});
        }}
    </script>
</body>
</html>'''
    
    return html_content

def main():
    # Set up paths
    user_results_dir = "results_qwen_user"
    assistant_results_dir = "results_qwen_assistant"
    output_file = "visualizer.html"
    
    print("Loading data from user results directory...")
    user_data = load_data(user_results_dir)
    print(f"Loaded {len(user_data)} user data files")
    
    print("Loading data from assistant results directory...")
    assistant_data = load_data(assistant_results_dir)
    print(f"Loaded {len(assistant_data)} assistant data files")
    
    print("Generating HTML visualizer...")
    html_content = generate_html(user_data, assistant_data)
    
    print(f"Writing to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Visualizer generated successfully: {output_file}")
    print(f"Open {output_file} in your browser to use the visualizer")

if __name__ == "__main__":
    main()