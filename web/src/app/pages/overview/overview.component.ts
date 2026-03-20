import { Component, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule } from '@angular/router';
import { ApiService, OverviewData, SignalFeedEntry } from '../../services/api.service';
import { Subscription, interval } from 'rxjs';
import { switchMap } from 'rxjs/operators';

@Component({
  selector: 'app-overview',
  standalone: true,
  imports: [CommonModule, RouterModule],
  template: `
    <div class="space-y-6 animate-fade-in">
      <h1 class="page-title">Overview</h1>

      <!-- Loading -->
      <div *ngIf="loading" class="flex items-center gap-3 text-gray-400">
        <svg class="animate-spin w-5 h-5" fill="none" viewBox="0 0 24 24">
          <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
          <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path>
        </svg>
        Loading data...
      </div>

      <!-- Error -->
      <div *ngIf="error" class="glass-card border-red-500/20 text-red-400">
        <p class="text-sm">{{ error }}</p>
      </div>

      <!-- KPI cards -->
      <div *ngIf="data" class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3 sm:gap-4">
        <div class="stat-card animate-slide-up" style="animation-delay: 0ms">
          <span class="stat-label">Feature Bars</span>
          <span class="stat-value">{{ fmt(data.kpis.feature_bars) }}</span>
        </div>
        <div class="stat-card animate-slide-up" style="animation-delay: 50ms">
          <span class="stat-label">PM Price Points</span>
          <span class="stat-value">{{ fmt(data.kpis.pm_prices) }}</span>
        </div>
        <div class="stat-card animate-slide-up" style="animation-delay: 100ms">
          <span class="stat-label">PM Events</span>
          <span class="stat-value">{{ fmt(data.kpis.pm_events) }}</span>
        </div>
        <div class="stat-card animate-slide-up" style="animation-delay: 150ms">
          <span class="stat-label">Scored Headlines</span>
          <span class="stat-value">{{ fmt(data.kpis.sentiment_scored) }}</span>
        </div>
        <div class="stat-card animate-slide-up" style="animation-delay: 200ms">
          <span class="stat-label">Latest Data</span>
          <span class="text-lg font-semibold text-white">{{ fmtTs(data.kpis.latest_ts) }}</span>
        </div>
      </div>

      <!-- Pipeline status -->
      <div *ngIf="data" class="glass-card animate-slide-up" style="animation-delay: 250ms">
        <h3 class="section-title mb-3">Pipeline Status</h3>
        <div class="flex flex-wrap gap-1.5 items-center">
          <span *ngFor="let s of data.pipeline_status"
                class="inline-block w-3 h-3 rounded-full transition-colors"
                [class]="statusDot(s.status)"
                [title]="s.table + ' (' + s.status + ')'">
          </span>
        </div>
        <div class="flex flex-wrap gap-3 mt-3 text-xs text-gray-500">
          <span class="flex items-center gap-1.5"><span class="w-2 h-2 rounded-full bg-emerald-500"></span> Fresh (24h)</span>
          <span class="flex items-center gap-1.5"><span class="w-2 h-2 rounded-full bg-amber-500"></span> Stale (1-2d)</span>
          <span class="flex items-center gap-1.5"><span class="w-2 h-2 rounded-full bg-red-500"></span> Old (&gt;2d)</span>
          <span class="flex items-center gap-1.5"><span class="w-2 h-2 rounded-full bg-gray-600"></span> Empty</span>
        </div>
      </div>

      <!-- Feature coverage -->
      <div *ngIf="data?.feature_coverage_7d as cov" class="glass-card animate-slide-up" style="animation-delay: 300ms">
        <h3 class="section-title mb-3">Feature Coverage (Last 7 Days)</h3>
        <div class="grid grid-cols-2 sm:grid-cols-5 gap-4">
          <div>
            <p class="text-xs text-gray-500">Bars</p>
            <p class="text-lg font-semibold text-white">{{ cov.bars }}</p>
          </div>
          <div>
            <p class="text-xs text-gray-500">IV</p>
            <p class="text-lg font-semibold text-white">{{ cov.iv }}</p>
          </div>
          <div>
            <p class="text-xs text-gray-500">PM</p>
            <p class="text-lg font-semibold text-white">{{ cov.pm }}</p>
          </div>
          <div>
            <p class="text-xs text-gray-500">Sentiment</p>
            <p class="text-lg font-semibold text-white">{{ cov.sentiment }}</p>
          </div>
          <div>
            <p class="text-xs text-gray-500">Equity</p>
            <p class="text-lg font-semibold text-white">{{ cov.equity }}</p>
          </div>
        </div>
      </div>

      <!-- Last pipeline run -->
      <div *ngIf="data" class="glass-card animate-slide-up" style="animation-delay: 350ms">
        <h3 class="section-title mb-3">Last Pipeline Run</h3>
        <pre class="text-xs text-gray-400 font-mono overflow-x-auto whitespace-pre-wrap leading-relaxed">{{ data.last_pipeline_line || 'No pipeline logs' }}</pre>
        <p *ngIf="data.sentiment_summary" class="text-sm text-gray-500 mt-3 pt-3 border-t border-white/[0.06]">
          Sentiment: {{ data.sentiment_summary }}
        </p>
      </div>

      <!-- Latest signals -->
      <div *ngIf="recentSignals.length > 0" class="glass-card animate-slide-up" style="animation-delay: 400ms">
        <div class="flex items-center justify-between mb-4">
          <h3 class="section-title">Latest Signals</h3>
          <a routerLink="/signals" class="text-xs text-blue-400 hover:text-blue-300 transition">View all &rarr;</a>
        </div>
        <p class="text-xs text-gray-500 mb-4">Recent position changes (SPY)</p>
        <div class="space-y-3">
          <div *ngFor="let entry of recentSignals.slice(0, 5)"
               class="rounded-xl border border-white/[0.06] bg-white/[0.02] p-4 hover:border-white/[0.1] transition">
            <div class="flex items-center justify-between mb-2">
              <span class="text-xs text-gray-500">{{ fmtTs(entry.ts) }} &middot; {{ entry.underlying }}</span>
              <span *ngIf="entry.outcome?.pnl != null"
                    class="text-xs font-medium"
                    [class.text-emerald-400]="(entry.outcome?.pnl ?? 0) >= 0"
                    [class.text-red-400]="(entry.outcome?.pnl ?? 0) < 0">
                PnL: {{ (entry.outcome?.pnl ?? 0).toFixed(2) }}
              </span>
            </div>
            <div *ngIf="entry.news.length > 0" class="flex items-start gap-2">
              <span class="badge badge-blue shrink-0 mt-0.5">News</span>
              <p class="text-sm text-gray-300 line-clamp-2">{{ entry.news[0].text }}</p>
            </div>
            <div *ngIf="entry.news.length === 0 && entry.pm_signals.length > 0" class="flex items-start gap-2">
              <span class="badge shrink-0 mt-0.5"
                    [class]="pmBadge(entry.pm_signals[0].platform)">{{ entry.pm_signals[0].platform || 'PM' }}</span>
              <p class="text-sm text-gray-300 line-clamp-2">
                {{ entry.pm_signals[0].event_name }}
                <span *ngIf="entry.pm_signals[0].probability != null" class="text-gray-500">
                  {{ (entry.pm_signals[0].probability! * 100).toFixed(0) }}%
                </span>
              </p>
            </div>
            <p *ngIf="entry.news.length === 0 && entry.pm_signals.length === 0"
               class="text-sm text-gray-500 italic">No source in window</p>
          </div>
        </div>
      </div>
    </div>
  `,
})
export class OverviewComponent implements OnInit, OnDestroy {
  data: OverviewData | null = null;
  recentSignals: SignalFeedEntry[] = [];
  loading = true;
  error: string | null = null;
  private subs: Subscription[] = [];

  constructor(private api: ApiService) {}

  ngOnInit() {
    this.api.getOverview().subscribe({
      next: d => { this.data = d; this.loading = false; },
      error: e => { this.error = e.message || 'Failed to load'; this.loading = false; },
    });

    this.api.getSignalFeed('SPY', 10, true).subscribe({
      next: d => this.recentSignals = d.feed,
      error: () => {},
    });

    // Auto-refresh every 60s
    this.subs.push(
      interval(60000).pipe(switchMap(() => this.api.getOverview())).subscribe({
        next: d => this.data = d,
      })
    );
  }

  ngOnDestroy() { this.subs.forEach(s => s.unsubscribe()); }

  fmt(n: number): string { return n?.toLocaleString() ?? '0'; }

  fmtTs(s: string | null): string {
    if (!s) return '\u2014';
    try {
      return new Date(s).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' });
    } catch { return s; }
  }

  statusDot(status: string): string {
    switch (status) {
      case 'fresh': return 'bg-emerald-500';
      case 'stale': return 'bg-amber-500';
      case 'old': return 'bg-red-500';
      default: return 'bg-gray-600';
    }
  }

  pmBadge(platform: string): string {
    return (platform || '').toLowerCase().includes('kalshi') ? 'badge badge-purple' : 'badge badge-yellow';
  }
}
