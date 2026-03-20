import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService, SignalFeedEntry } from '../../services/api.service';

@Component({
  selector: 'app-trade-impact',
  standalone: true,
  imports: [CommonModule, FormsModule],
  template: `
    <div class="space-y-6 animate-fade-in">
      <div class="flex flex-col gap-3">
        <h1 class="page-title">Trade Impact</h1>
        <p class="text-sm text-gray-500">Signal &rarr; action &rarr; outcome for bars where the agent changed position.</p>
        <div class="flex items-center gap-3 flex-wrap">
          <input [(ngModel)]="underlying" (keyup.enter)="load()" placeholder="SPY"
                 class="input-field w-24 text-sm !py-2" />
          <label class="flex items-center gap-2 text-sm text-gray-400 cursor-pointer select-none">
            <input type="checkbox" [(ngModel)]="showAll" (change)="load()"
                   class="w-4 h-4 rounded border-gray-600 bg-white/[0.04] text-blue-500 focus:ring-blue-500/30" />
            Show all bars with any position change
          </label>
        </div>
      </div>

      <div *ngIf="loading" class="flex items-center gap-3 text-gray-400">
        <svg class="animate-spin w-5 h-5" fill="none" viewBox="0 0 24 24">
          <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
          <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path>
        </svg>
        Loading...
      </div>

      <div *ngIf="error" class="glass-card border-red-500/20 text-red-400 text-sm">{{ error }}</div>

      <div *ngIf="!loading && !error">
        <p class="text-sm text-gray-400 mb-4">
          Found <strong class="text-white">{{ bars.length }}</strong> bars with position changes.
        </p>

        <div *ngIf="bars.length === 0" class="glass-card text-center py-12">
          <p class="text-gray-500">No significant position changes.</p>
          <p class="text-sm text-gray-600 mt-1">Try more bars, a different underlying, or enable the checkbox above.</p>
        </div>

        <div *ngIf="bars.length > 0" class="space-y-4">
          <div *ngFor="let entry of bars.slice(0, 20); let i = index"
               class="glass-card-hover animate-slide-up"
               [style.animation-delay]="(i * 40) + 'ms'">

            <!-- Header -->
            <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-3">
              <span class="text-sm font-medium text-white">
                {{ fmtTs(entry.ts) }}
                &mdash; &Delta;delta={{ (entry.action!.delta_change ?? 0).toFixed(0) }},
                &Delta;vega={{ (entry.action!.vega_change ?? 0).toFixed(0) }}
              </span>
              <span class="text-sm font-semibold tabular-nums"
                    [class.text-emerald-400]="(entry.outcome?.pnl ?? 0) >= 0"
                    [class.text-red-400]="(entry.outcome?.pnl ?? 0) < 0">
                PnL: {{ fmtNum(entry.outcome?.pnl) }}
              </span>
            </div>

            <!-- PM signals -->
            <div *ngIf="entry.pm_signals.length > 0" class="flex flex-wrap gap-2 mb-3">
              <span *ngFor="let pm of entry.pm_signals"
                    class="badge text-xs"
                    [class]="pmBadge(pm.platform)">
                {{ pm.platform || 'PM' }}: {{ pm.event_name }}
                <span *ngIf="pm.probability != null"> {{ (pm.probability! * 100).toFixed(1) }}%</span>
                <span *ngIf="pm.delta_1h != null">
                  ({{ pm.delta_1h! >= 0 ? '+' : '' }}{{ (pm.delta_1h! * 100).toFixed(1) }}% 1h)
                </span>
              </span>
            </div>

            <!-- Outcome -->
            <div class="text-xs text-gray-400 space-y-1">
              <p>
                <strong class="text-gray-300">Outcome:</strong>
                PnL this bar: <strong class="text-white">{{ fmtNum(entry.outcome?.pnl) }}</strong>
              </p>
              <p *ngIf="entry.iv?.atm_iv_30d != null">
                ATM IV 30d: {{ (entry.iv.atm_iv_30d! * 100).toFixed(1) }}%
              </p>
            </div>

            <!-- News -->
            <div *ngIf="entry.news.length > 0" class="mt-4 pt-4 border-t border-white/[0.06]">
              <p class="text-[10px] font-bold uppercase tracking-[0.15em] text-gray-600 mb-3">
                News that may have triggered this move
              </p>
              <div *ngFor="let h of entry.news"
                   class="rounded-xl bg-white/[0.03] border border-white/[0.04] p-3.5 mb-2.5">
                <p class="text-sm text-gray-200 leading-relaxed">{{ h.text || '\u2014' }}</p>
                <div class="mt-2 flex items-center gap-3 flex-wrap text-xs">
                  <span class="text-gray-500">{{ h.source || 'Unknown' }}</span>
                  <span [class]="sentimentClass(h.compound ?? 0)">{{ sentimentLabel(h.compound ?? 0) }}</span>
                  <span *ngIf="h.ts" class="text-gray-500">{{ fmtTs(h.ts!) }}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  `,
})
export class TradeImpactComponent implements OnInit {
  underlying = 'SPY';
  showAll = false;
  bars: SignalFeedEntry[] = [];
  loading = true;
  error: string | null = null;

  constructor(private api: ApiService) {}

  ngOnInit() { this.load(); }

  load() {
    this.loading = true;
    this.error = null;
    this.api.getTradeImpact(this.underlying, 200, this.showAll).subscribe({
      next: d => { this.bars = d.bars as any; this.loading = false; },
      error: e => { this.error = e.message; this.loading = false; },
    });
  }

  fmtTs(s: string | null | undefined): string {
    if (!s) return '\u2014';
    try { return new Date(s).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' }); }
    catch { return s; }
  }

  fmtNum(x: number | null | undefined): string {
    if (x == null) return '\u2014';
    return Number(x).toFixed(2);
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
    return (platform || '').toLowerCase().includes('kalshi') ? 'badge-purple' : 'badge-yellow';
  }
}
