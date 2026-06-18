/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 *
 * Derived from mlsyscourse/slides-modern-gpu-programming
 * (data-layout/site/viz-base.js). Shared behavior for the TIRx
 * layout visualization; not derived from any third-party demo.
 */

// Shared behavior for all viz HTMLs
document.addEventListener('DOMContentLoaded', function() {
  var p = new URLSearchParams(location.search);
  if (p.has('notitle')) document.body.classList.add('notitle');

  // Forward arrow keys to parent (reveal.js) when embedded
  if (window.parent !== window) {
    document.addEventListener('keydown', function(e) {
      if ([37, 38, 39, 40, 27, 32].indexOf(e.keyCode) !== -1) {
        // Left, Up, Right, Down, Escape, Space
        window.parent.postMessage({ type: 'revealKey', keyCode: e.keyCode }, '*');
      }
    });
  }
});
