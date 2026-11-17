import { Component, type ReactNode } from 'react';

export class ViewportErrorBoundary extends Component<
  { children: ReactNode; hidden?: boolean },
  { failed: boolean; attempt: number }
> {
  state = { failed: false, attempt: 0 };

  static getDerivedStateFromError() {
    return { failed: true };
  }

  render() {
    if (this.state.failed) {
      return (
        <main
          className="editor-viewport"
          style={this.props.hidden ? { display: 'none' } : undefined}
        >
          <div className="viewport-loading-overlay" role="alert">
            <div className="loading-pill">
              <span>The viewport could not start. Check that WebGL2 is enabled.</span>
              <button
                type="button"
                className="toolbar-button"
                onClick={() => this.setState({ failed: false, attempt: this.state.attempt + 1 })}
              >
                Retry Viewport
              </button>
            </div>
          </div>
        </main>
      );
    }
    return (
      <div key={this.state.attempt} style={{ display: 'contents' }}>
        {this.props.children}
      </div>
    );
  }
}
