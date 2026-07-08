# Mnemodrome Admin UI - Static Files

This directory contains the static web UI files for the Mnemodrome Admin Interface.

## File Structure

```
src/static/
├── index.html      # Home dashboard page
├── config.html     # Configuration editor page
├── health.html     # Health monitoring dashboard
├── events.html     # Event log viewer page
├── style.css       # Shared stylesheet for all pages
└── app.js          # Shared JavaScript utilities and API client
```

## File Purposes

### HTML Pages

- **index.html**: Dashboard home page with system overview, quick stats, and navigation to all admin sections
- **config.html**: Configuration editor with live validation, category-based forms, and undo/export functionality
- **health.html**: Real-time health monitoring dashboard with auto-refresh and visual status indicators
- **events.html**: Event log viewer with filtering, pagination, and cleanup capabilities

### Shared Resources

- **style.css**: Professional admin UI styling with responsive design, color-coded status indicators, and mobile support
- **app.js**: Shared JavaScript utilities including API client, validation, error handling, and UI helpers

## Design Principles

1. **Vanilla JavaScript**: No external dependencies - pure HTML, CSS, and JavaScript
2. **Responsive Design**: Mobile-first approach with breakpoints for tablet and desktop
3. **Accessibility**: Semantic HTML5 with ARIA labels and keyboard navigation
4. **Security**: HTTP Basic Auth integration with browser-native authentication
5. **User Experience**: Clear error messages, loading states, and success feedback

## Browser Compatibility

Tested and supported on:
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## API Integration

All pages communicate with the FastAPI backend via these endpoints:
- `GET /admin/config` - Read configuration
- `POST /admin/config` - Update configuration
- `POST /admin/config/undo` - Undo last change
- `GET /admin/health` - System health status
- `GET /admin/events` - Query events
- `POST /admin/events/cleanup` - Trigger cleanup
- `GET /admin/health-check` - Simple health check (no auth)
