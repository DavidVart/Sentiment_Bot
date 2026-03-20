import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService, SignalFeedEntry } from '../../services/api.service';

@Component({
  selector: 'app-signals',
  standalone: true,
  imports: [CommonModule, FormsModule],
  template: `
    <div class="space-y-6 animate-fade-in">
      <!-- Header -->
      <div class="flex flex-col sm:flex-row sm:items-center gap-4">
        <h1 class="page-title">Signal Feed</h1>
        <div class="flex items-center gap-3 flex-wrap">
          <input [(ngModel)]="underlying" (keyup.enter)="load()" placeholder="SPY"
                 class="input-field w-24 text-sm !py-2" />
          <label class="flex items-center gap-2 text-sm text-gray-400 cursor-pointer select-none">
            <input type="checkbox" [(ngModel)]="positionChangesOnly" (change)="load()"
                   class="w-4 h-4 rounded border-gray-600 bg-white/[0.04] text-blue-500 focus:ring-blue-500/30" />
            Position changes only
          </label>
          <button (click)="load()" class="btn-secondary !py-2 text-sm">Refresh</button>
        </div>
      </div>

      <p class="text-sm text-gray-500">
        Source &rarr; Stock &rarr; Position. News, Polymarket/Kalshi events, and agent action/outcome per bar.
      </p>

      <!-- Loading -->
      <div *ngIf="loading" class="flex items-center gap-3 text-gray-400">
        <svg class="animate-spin w-5 h-5" fill="none" viewBox="0 0 24 24">
          <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
          <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path>
        </svg>
        Loading signals...
      </div>

      <!-- Error -->
      <div *ngIf="error" class="glass-card border-red-500/20 text-red-400 text-sm">{{ error }}</div>

      <!-- Empty -->
      <div *ngIf="!loading && feed.length === 0" class="glass-card text-center py-12">
        <p class="text-gray-500">No signal feed data available.</p>
        <p class="text-sm text-gray-600 mt-1">Run the pipeline with step 4 (PM backfill) and snapshot.</p>
      </div>

      <!-- Feed entries -->
      <div *ngIf="!loading && feed.length > 0" class="space-y-4">
        <div *ngFor="let entry of feed.slice(0, 50); let i = index"
             class="glass-card-hover animate-slide-up"
             [style.animation-delay]="(i * 30) + 'ms'">

          <div class="flex items-center justify-between mb-3">
            <span class="text-sm font-medium text-white">{{ fmtTs(entry.ts) }} &mdash; {{ entry.underlying }}</span>
            <span *ngIf="entry.outcome?.pnl != null"
                  class="text-sm font-semibold"
                  [class.text-emerald-400]="(entry.outcome?.pnl ?? 0) >= 0"
                  [class.text-red-400]="(entry.outcome?.pnl ?? 0) < 0">
              PnL {{ (entry.outcome?.pnl ?? 0).toFixed(2) }}
            </span>
          </div>

          <!-- News sources -->
          <div *ngIf="entry.news.length > 0" class="mb-3">
            <p class="text-[10px] font-bold uppercase tracking-[0.15em] text-gray-600 mb-2">Sources</p>
            <div *ngFor="let n of entry.news" class="flex items-start gap-2.5 mb-2.5">
              <span class="badge badge-blue shrink-0 mt-0.5">News</span>
              <div class="min-w-0">
                <p class="text-sm text-gray-300 leading-relaxed">{{ n.text || '\u2014' }}</p>
                <p class="text-xs text-gray-500 mt-1">
                  {{ n.source }}
                  &middot;
                  <span [class]="sentimentClass(n.compound ?? 0)">{{ sentimentLabel(n.compound ?? 0) }}</span>
                  <span *ngIf="n.compound != null"> ({{ n.compound!.toFixed(2) }})</span>
                </p>
              </div>
            </div>
          </div>

          <!-- PM signals -->
          <div *ngIf="entry.pm_signals.length > 0">
            <p class="text-[10px] font-bold uppercase tracking-[0.15em] text-gray-600 mb-2">
              {{ entry.news.length > 0 ? 'Prediction Markets' : 'Sources' }}
            </p>
            <div *ngFor="let pm of entry.pm_signals" class="flex items-start gap-2.5 mb-2.5">
              <span class="badge shrink-0 mt-0.5" [class]="pmBadge(pm.platform)">{{ pm.platform || 'PM' }}</span>
              <div class="min-w-0">
                <p class="text-sm text-gray-300">{{ pm.event_name || 'Event' }}</p>
                <p class="text-xs text-gray-500 mt-0.5">
                  Prob: {{ pm.probability != null ? (pm.probability! * 100).toFixed(1) + '%' : '\u2014' }}
                  <span *ngIf="pm.delta_1h != null">
                    ({{ pm.delta_1h! >= 0 ? '+' : '' }}{{ (pm.delta_1h! * 100).toFixed(1) }}% 1h)
                  </span>
                </p>
              </div>
            </div>
          </div>

          <p *ngIf="entry.news.length === 0 && entry.pm_signals.length === 0"
             class="text-xs text-gray-500 italic">No news or PM signals for this bar.</p>

          <!-- Action / IV footer -->
          <div class="mt-4 pt-3 border-t border-white/[0.06] flex flex-wrap gap-4 text-sm">
            <span *ngIf="entry.action" class="text-gray-400">
              <strong class="text-gray-300">Position</strong>
              &Delta;delta={{ (entry.action.delta_change ?? 0).toFixed(0) }}
              &Delta;vega={{ (entry.action.vega_change ?? 0).toFixed(0) }}
            </span>
            <span *ngIf="entry.iv?.atm_iv_30d != null" class="text-gray-400">
              <strong class="text-gray-300">IV</strong>
              ATM 30d: {{ (entry.iv.atm_iv_30d! * 100).toFixed(1) }}%
            </span>
          </div>
        </div>
      </div>
    </div>
  `,
})
export class SignalsComponent implements OnInit {
  underlying = 'SPY';
  positionChangesOnly = false;
  feed: SignalFeedEntry[] = [];
  loading = true;
  error: string | null = null;

  constructor(private api: ApiService) {}

  ngOnInit() { this.load(); }

  load() {
    this.loading = true;
    this.error = null;
    this.api.getSignalFeed(this.underlying, 100, this.positionChangesOnly).subscribe({
      next: d => { this.feed = d.feed; this.loading = false; },
      error: e => { this.error = e.message; this.loading = false; },
    });
  }

  fmtTs(s: string | null): string {
    if (!s) return '\u2014';
    try { return new Date(s).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' }); }
    catch { return s; }
  }

  sentimentLabel(c: number): string {
    if (c >= 0.05) return 'Positive';
    if (c <= -0.05) return 'Negative';
    return 'Neutral';
  }

  sentimentClass(c: number): string {
    if (c >= 0.05) return 'text-emerald-400';
    if (c <= -0.05) return 'text-red-400';
    return 'text-gray-400';
  }

  pmBadge(platform: string): string {
    return (platform || '').toLowerCase().includes('kalshi') ? 'badge badge-purple' : 'badge badge-yellow';
  }
}
