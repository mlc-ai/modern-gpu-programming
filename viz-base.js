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
