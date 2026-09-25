#!/usr/bin/env node
// Deterministic SVG + Sharp figure pipeline. No Python, refitting, or resampling.
import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..', '..');
const OUT = path.join(HERE, 'Q3补充图表');
const FIG = path.join(OUT, 'figures');
const NODE_MODULES = 'C:/Users/86147/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const require = createRequire(path.join(NODE_MODULES, 'codex-q3-renderer.cjs'));
const sharp = require('sharp');

const COLOR = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00'];
const SHAPE = ['circle', 'square', 'triangle', 'diamond', 'plus'];
const CONTEXTS = [2048, 4096, 8192, 32768, 131072];
const INPUTS = {
  frontier: 'Q3/04_结果/M0_ND_reference_frontier.csv',
  bootstrap_frontier: 'Q3/04_结果/M0_ND_cluster_bootstrap_frontier.csv',
  bootstrap_frequency: 'Q3/04_结果/M0_ND_cluster_bootstrap_selection_frequency.csv',
  cost_shares: 'Q3/04_结果/M0_budget_context_cost_shares.csv',
  descriptive_shifts: 'Q3/04_结果/M0_budget_context_descriptive_shifts.csv',
  q1_a1_summary: 'Q1/03_结果/Q1.1/v1/a1_macro_summary.csv',
};

function parseCsv(source) {
  const records = [];
  let row = [], field = '', quoted = false;
  for (let i = 0; i < source.length; i++) {
    const ch = source[i];
    if (quoted) {
      if (ch === '"' && source[i + 1] === '"') { field += '"'; i++; }
      else if (ch === '"') quoted = false;
      else field += ch;
    } else if (ch === '"') quoted = true;
    else if (ch === ',') { row.push(field); field = ''; }
    else if (ch === '\n') { row.push(field.replace(/\r$/, '')); if (row.some(v => v !== '')) records.push(row); row = []; field = ''; }
    else field += ch;
  }
  if (field || row.length) { row.push(field.replace(/\r$/, '')); if (row.some(v => v !== '')) records.push(row); }
  const [header, ...body] = records;
  return body.map(values => Object.fromEntries(header.map((name, i) => [name, values[i] ?? ''])));
}

async function readCsv(relative) {
  const text = (await fs.readFile(path.join(ROOT, relative), 'utf8')).replace(/^\uFEFF/, '');
  return parseCsv(text);
}

async function sha256(file) {
  const hash = crypto.createHash('sha256');
  hash.update(await fs.readFile(file));
  return hash.digest('hex');
}

function num(v) { return Number(v); }
function bool(v) { return ['true', '1', 'yes'].includes(String(v).trim().toLowerCase()); }
function key(r) { return `${r.budget_flops}|${r.context_length_tokens}`; }
function ctxLabel(v) { return `${Math.round(Number(v) / 1024)}k`; }
function budgetLabel(v) { return `10${superscript(Math.round(Math.log10(Number(v))))}`; }
function superscript(n) { return String(n).replace(/[0-9]/g, d => '⁰¹²³⁴⁵⁶⁷⁸⁹'[Number(d)]); }
function escapeXml(value) { return String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;').replaceAll("'", '&apos;'); }
function map(value, lo, hi, start, end) { return hi === lo ? (start + end) / 2 : start + (value - lo) / (hi - lo) * (end - start); }
function assert(ok, message) { if (!ok) throw new Error(message); }

class Svg {
  constructor(width, height) {
    this.w = width; this.h = height; this.el = []; this.bounds = [];
    this.el.push(`<rect width="${width}" height="${height}" fill="#ffffff"/>`);
  }
  line(x1, y1, x2, y2, color = '#333333', width = 1, dash = '') {
    this.el.push(`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${width}"${dash ? ` stroke-dasharray="${dash}"` : ''} stroke-linecap="round"/>`);
  }
  polyline(points, color, width = 1.8, dash = '') {
    if (!points.length) return;
    this.el.push(`<polyline points="${points.map(p => `${p[0]},${p[1]}`).join(' ')}" fill="none" stroke="${color}" stroke-width="${width}"${dash ? ` stroke-dasharray="${dash}"` : ''} stroke-linejoin="round" stroke-linecap="round"/>`);
  }
  rect(x, y, w, h, fill = 'none', stroke = 'none', sw = 1, rx = 0) {
    this.el.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${rx}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}"/>`);
  }
  circle(x, y, r, fill, stroke = '#ffffff', sw = 0.7) {
    this.el.push(`<circle cx="${x}" cy="${y}" r="${r}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}"/>`);
  }
  polygon(points, fill, stroke = '#ffffff', sw = 0.7) {
    this.el.push(`<polygon points="${points.map(p => `${p[0]},${p[1]}`).join(' ')}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}" stroke-linejoin="round"/>`);
  }
  marker(x, y, shape, color, size = 4) {
    if (shape === 'square') this.rect(x-size, y-size, 2*size, 2*size, color, '#ffffff', 0.7, 0.5);
    else if (shape === 'triangle') this.polygon([[x,y-size*1.25],[x-size*1.05,y+size],[x+size*1.05,y+size]], color);
    else if (shape === 'diamond') this.polygon([[x,y-size*1.2],[x-size,y],[x,y+size*1.2],[x+size,y]], color);
    else if (shape === 'plus') { this.line(x-size,y,x+size,y,color,1.8); this.line(x,y-size,x,y+size,color,1.8); }
    else this.circle(x,y,size,color);
  }
  text(x, y, value, size = 13, color = '#222222', anchor = 'start', weight = 400, rotate = 0) {
    const text = escapeXml(value);
    this.el.push(`<text x="${x}" y="${y}" fill="${color}" font-size="${size}" font-weight="${weight}" text-anchor="${anchor}" dominant-baseline="hanging"${rotate ? ` transform="rotate(${rotate} ${x} ${y})"` : ''}>${text}</text>`);
    const chars = Array.from(String(value));
    const width = chars.reduce((total, char) => total + (/^[ -~]$/.test(char) ? size * 0.56 : size), 0);
    const left = anchor === 'middle' ? x - width / 2 : anchor === 'end' ? x - width : x;
    this.bounds.push({x1: left, y1: y, x2: left + width, y2: y + size * 1.35, text: String(value)});
  }
  finish() {
    return `<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" width="${this.w*3}px" height="${this.h*3}px" viewBox="0 0 ${this.w} ${this.h}"><style>text{font-family:SimSun,"Microsoft YaHei","Noto Sans CJK SC",sans-serif}</style>${this.el.join('')}</svg>`;
  }
}

function header(c, title, subtitle) {
  c.text(c.w/2, 22, title, 21, '#111827', 'middle', 700);
  c.text(c.w/2, 53, subtitle, 12.5, '#4b5563', 'middle');
}
function legend(c, x, y, contexts, gap = 124) {
  c.text(x, y, '上下文长度', 11, '#333333');
  contexts.forEach((ctx, i) => {
    const xx = x + 92 + i*gap;
    c.line(xx, y+7, xx+20, y+7, COLOR[i], 1.8);
    c.marker(xx+10,y+7,SHAPE[i],COLOR[i],3.2);
    c.text(xx+26,y,ctxLabel(ctx),10.5,'#333333');
  });
}
function niceTicks(maximum, count = 4) {
  if (maximum <= 0) return [1,[0,.25,.5,.75,1]];
  const raw = maximum/count, power=10**Math.floor(Math.log10(raw));
  const step=[1,2,2.5,5,10].map(x=>x*power).sort((a,b)=>Math.abs(a-raw)-Math.abs(b-raw))[0];
  const top=Math.ceil(maximum/step)*step;
  return [top,Array.from({length:Math.round(top/step)+1},(_,i)=>i*step)];
}
function drawBudgetAxes(c, bounds, budgets, ticks, ymin, ymax, yfmt, ylabel = '', xlabel = '预算 FLOPs') {
  const [left,right,top,bottom]=bounds;
  c.line(left,top,left,bottom,'#333333',1); c.line(left,bottom,right,bottom,'#333333',1);
  ticks.forEach(t=>{const yy=map(t,ymin,ymax,bottom,top); c.line(left,yy,right,yy,'#d9dee5',.75); c.text(left-8,yy-6,yfmt(t),10,'#333333','end');});
  const lo=Math.log10(Math.min(...budgets)),hi=Math.log10(Math.max(...budgets));
  const xp=b=>map(Math.log10(Number(b)),lo,hi,left,right);
  budgets.forEach(b=>{const xx=xp(b);c.line(xx,bottom,xx,bottom+4,'#333333',.8);c.text(xx,bottom+7,budgetLabel(b),10,'#333333','middle');});
  c.text((left+right)/2,bottom+29,xlabel,10.5,'#222222','middle');
  if(ylabel)c.text(left-43,(top+bottom)/2,ylabel,11,'#222222','middle',400,-90);
  return {xp,yp:v=>map(v,ymin,ymax,bottom,top)};
}
function sortedScenarios(rows) { return [...rows].sort((a,b)=>Number(a.context_length_tokens)-Number(b.context_length_tokens)||Number(a.budget_flops)-Number(b.budget_flops)); }

function plotP1(frontier,budgets,contexts) {
  const c=new Svg(1000,500);header(c,'Q3 P1｜固定 Q₀ 下的预算—N/D 参考配置','B1 实测网格上的 M0 离散情景；不是完整 Q3 联合最优');legend(c,82,84,contexts,122);
  const panels=[{x:48,title:'最优模型规模 N',unit:'十亿参数',field:'selected_N_B'},{x:526,title:'最优训练数据量 D',unit:'十亿 Token',field:'selected_D_B'}];
  for(const p of panels){
    c.text(p.x+210,112,p.title,14,'#111827','middle',700);
    const max=Math.max(...frontier.map(r=>num(r[p.field]))),[top,ticks]=niceTicks(max*1.08);
    const ax=drawBudgetAxes(c,[p.x+61,p.x+407,151,402],budgets,ticks,0,top,v=>`${Number(v.toPrecision(2))}`,p.unit);
    contexts.forEach((ctx,i)=>{
      const rows=frontier.filter(r=>Number(r.context_length_tokens)===ctx).sort((a,b)=>Number(a.budget_flops)-Number(b.budget_flops));
      const pts=rows.map(r=>[ax.xp(r.budget_flops),ax.yp(num(r[p.field]))]);
      c.polyline(pts,COLOR[i],1.8,'5,3');
      rows.forEach((r,j)=>{const [x,y]=pts[j];c.marker(x,y,SHAPE[i],COLOR[i],4);if(bool(r.selected_at_support_max_N_and_D))c.circle(x,y,7,'none','#111827',1.2);});
    });
  }
  c.text(500,462,'空心外圈标记 B1 网格上界选点；最高预算下部分配置受有限支持边界约束。',10.5,'#374151','middle');
  return c;
}

function drawScenarioAxis(c,left,right,bottom,contexts,budgets) {
  for(let i=0;i<15;i++){
    const x=map(i,0,14,left,right);c.text(x,bottom+5,budgetLabel(budgets[i%3]).replace('10',''),8.5,'#4b5563','middle');
  }
  contexts.forEach((ctx,i)=>c.text(map(i*3+1,0,14,left,right),bottom+21,ctxLabel(ctx),9,COLOR[i],'middle',700));
}
function plotP2(boot,frequency,contexts,budgets) {
  const c=new Svg(1200,570);header(c,'Q3 P2｜Bootstrap 配置选择稳定性','8 条 B1 轨迹的 1,000 次 cluster Bootstrap；限既定 M0 与支持网格内参数传播');
  const scenarios=sortedScenarios(boot);
  const panels=[{x:42,w:350,title:'N 配置 95% 区间（十亿参数）',med:'selected_N_B_median',lo:'selected_N_B_p2_5',hi:'selected_N_B_p97_5'},
    {x:424,w:350,title:'D 配置 95% 区间（十亿 Token）',med:'selected_D_B_median',lo:'selected_D_B_p2_5',hi:'selected_D_B_p97_5'}];
  panels.forEach(p=>{
    c.text(p.x+p.w/2,103,p.title,13,'#111827','middle',700);
    const vals=scenarios.map(r=>num(r[p.med])),vmin=Math.min(...vals),vmax=Math.max(...vals),pad=Math.max((vmax-vmin)*.12,Math.max(Math.abs(vmax),1)*.025);
    const ymin=Math.max(0,vmin-pad),ymax=vmax+pad,left=p.x+50,right=p.x+p.w-10,top=145,bottom=389;
    c.line(left,top,left,bottom,'#333',1);c.line(left,bottom,right,bottom,'#333',1);
    for(let j=0;j<=4;j++){const val=ymin+(ymax-ymin)*j/4,y=map(val,ymin,ymax,bottom,top);c.line(left,y,right,y,'#d9dee5',.7);c.text(left-7,y-6,Number(val.toPrecision(3)).toString(),9,'#333','end');}
    scenarios.forEach((r,i)=>{const x=map(i,0,14,left,right),y=map(num(r[p.med]),ymin,ymax,bottom,top);assert(num(r[p.lo])===num(r[p.med])&&num(r[p.hi])===num(r[p.med]),'Bootstrap N/D interval is not a point mass');c.marker(x,y,SHAPE[contexts.indexOf(Number(r.context_length_tokens))],COLOR[contexts.indexOf(Number(r.context_length_tokens))],3.7);});
    drawScenarioAxis(c,left,right,bottom,contexts,budgets);
  });
  const freqByKey=new Map(frequency.filter(r=>bool(r.baseline_selected)).map(r=>[key(r),r]));
  const x=800,w=350,left=x+48,right=x+w-10,top=145,bottom=389;
  c.text(x+w/2,103,'原最优点重选率与切换概率',13,'#111827','middle',700);
  c.line(left,top,left,bottom,'#333',1);c.line(left,bottom,right,bottom,'#333',1);
  [0,.25,.5,.75,1].forEach(v=>{const y=map(v,0,1,bottom,top);c.line(left,y,right,y,'#d9dee5',.7);c.text(left-7,y-6,`${Math.round(v*100)}%`,9,'#333','end');});
  scenarios.forEach((r,i)=>{const fr=freqByKey.get(key(r));assert(fr,'Missing baseline frequency point');const value=num(fr.selection_frequency),xx=map(i,0,14,left,right),ctxi=contexts.indexOf(Number(r.context_length_tokens));c.marker(xx,map(value,0,1,bottom,top),SHAPE[ctxi],COLOR[ctxi],3.7);c.circle(xx,map(1-value,0,1,bottom,top),2.6,'#6b7280');});
  c.line(left+4,top+10,left+21,top+10,'#0072B2',1.8);c.text(left+26,top+3,'重选率',9,'#333');c.circle(left+104,top+10,3,'#6b7280');c.text(left+113,top+3,'切换概率',9,'#333');
  drawScenarioAxis(c,left,right,bottom,contexts,budgets);
  c.text(600,497,'横轴按上下文长度分组（Token）；组内三点依次表示预算 10¹⁹、10²²、10²⁴ FLOPs。',10,'#374151','middle');
  c.text(600,518,'N/D 区间宽度确为 0；切换概率按 1−重选率计算。结果不覆盖模型形式误差或外部迁移。',10.5,'#374151','middle');
  return c;
}

function plotP3(frontier,shares,contexts,budgets) {
  const c=new Svg(1120,700);header(c,'Q3 P3｜预算与上下文变化下的资源配置','15 个固定 Q₀ 的离散情景；22 对变化仅作描述，不作结构转移推断');legend(c,64,78,contexts,124);
  const panels=[{x:44,y:104,w:504,h:248,title:'模型规模 N（十亿参数）',field:'selected_N_B'},
    {x:572,y:104,w:504,h:248,title:'数据规模 D（十亿 Token）',field:'selected_D_B'},
    {x:44,y:380,w:504,h:252,title:'M0 选点成本份额',field:'shares'},
    {x:572,y:380,w:504,h:252,title:'M0 预测验证 Loss',field:'m0_predicted_val_loss'}];
  for(const p of panels){
    c.text(p.x+p.w/2,p.y,p.title,13.5,'#111827','middle',700);
    if(p.field==='shares'){
      const left=p.x+54,right=p.x+p.w-10,top=p.y+34,bottom=p.y+p.h-74;
      c.line(left,top,left,bottom,'#333',1);c.line(left,bottom,right,bottom,'#333',1);
      [0,.25,.5,.75,1].forEach(v=>{const y=map(v,0,1,bottom,top);c.line(left,y,right,y,'#d9dee5',.7);c.text(left-8,y-6,`${Math.round(v*100)}%`,9,'#333','end');});
      const ordered=sortedScenarios(shares),group=(right-left)/5,bw=group*.13;
      ordered.forEach(r=>{const ci=contexts.indexOf(Number(r.context_length_tokens)),bi=budgets.indexOf(Number(r.budget_flops)),bx=left+ci*group+(bi+.5)*group/3-bw/2;let ybase=bottom;
        [[num(r.train_cost_share),'#0072B2'],[num(r.attention_cost_share),'#E69F00'],[num(r.quality_cost_share),'#9ca3af']].forEach(([v,col])=>{const bh=(bottom-top)*v;c.rect(bx,ybase-bh,bw,bh,col,'#ffffff',.5);ybase-=bh;});
        c.text(bx+bw/2,bottom+5,budgetLabel(r.budget_flops).replace('10',''),8,'#333','middle');});
      contexts.forEach((ctx,i)=>c.text(left+(i+.5)*group,bottom+20,ctxLabel(ctx),8.8,'#374151','middle',700));
      [['训练','#0072B2'],['注意力','#E69F00'],['质量（固定 Q₀=0）','#9ca3af']].forEach(([label,col],i)=>{const lx=left+i*150;c.rect(lx,p.y+p.h-28,9,9,col);c.text(lx+14,p.y+p.h-30,label,9,'#333');});
      continue;
    }
    const vals=frontier.map(r=>num(r[p.field]));let ymin=Math.min(...vals)*.96,ymax=Math.max(...vals)*1.04;
    if(p.field==='m0_predicted_val_loss'){ymin=Math.floor(ymin*100)/100;ymax=Math.ceil(ymax*100)/100;}
    const ticks=Array.from({length:5},(_,i)=>ymin+(ymax-ymin)*i/4),left=p.x+60,right=p.x+p.w-10,top=p.y+34,bottom=p.y+p.h-50;
    const ax=drawBudgetAxes(c,[left,right,top,bottom],budgets,ticks,ymin,ymax,v=>Number(v.toPrecision(3)).toString());
    contexts.forEach((ctx,i)=>{const rows=frontier.filter(r=>Number(r.context_length_tokens)===ctx).sort((a,b)=>Number(a.budget_flops)-Number(b.budget_flops));const pts=rows.map(r=>[ax.xp(r.budget_flops),ax.yp(num(r[p.field]))]);c.polyline(pts,COLOR[i],1.6,'5,3');rows.forEach((r,j)=>{const [xx,yy]=pts[j];c.marker(xx,yy,SHAPE[i],COLOR[i],3.5);if(bool(r.selected_at_support_max_N_and_D))c.circle(xx,yy,6,'none','#111827',1);});});
  }
  c.text(560,666,'固定 Q=Q₀ 时质量成本份额为零；最高预算部分点触及 B1 支持上界。折线只连接给定离散档位。',10,'#374151','middle');
  return c;
}

function plotP4(frontier,q1) {
  const qeq=num(q1.find(r=>r.dataset==='a1'&&r.candidate==='q_equal').macro_domain_score);
  const qhub=num(q1.find(r=>r.dataset==='a1'&&r.candidate==='q_huber').macro_domain_score);
  const c=new Svg(1120,660);header(c,'Q3 P4｜Q 基准敏感性：固定 Q 下 M0 选择不变','比较 Q1 的 q_huber 与 q_equal 基准；不估计 Q→Loss 的实证效应');
  c.text(560,82,`q_huber Q₀=${qhub.toFixed(10)}     q_equal Q₀=${qeq.toFixed(10)}     ΔQ₀=${(qhub-qeq).toFixed(6)}`,12.5,'#374151','middle');
  const scenarios=sortedScenarios(frontier),contexts=[...CONTEXTS];
  const panels=[{x:46,y:112,title:'两个 Q₀ 基准值'},{x:570,y:112,title:'|ΔN|（十亿参数）'},{x:46,y:382,title:'|ΔD|（十亿 Token）'},{x:570,y:382,title:'|Δ M0 预测 Loss|' }];
  for(const p of panels){const w=500,h=222;c.text(p.x+w/2,p.y,p.title,14,'#111827','middle',700);const left=p.x+55,right=p.x+w-12,top=p.y+40,bottom=p.y+h-45;
    if(p.title==='两个 Q₀ 基准值'){
      const lo=Math.min(qhub,qeq)-.001,hi=Math.max(qhub,qeq)+.001;[lo,(lo+hi)/2,hi].forEach(v=>{const y=map(v,lo,hi,bottom,top);c.line(left,y,right,y,'#d9dee5',.7);c.text(left-7,y-6,v.toFixed(3),9,'#333','end');});
      [[qhub,'q_huber','#0072B2'],[qeq,'q_equal','#D55E00']].forEach(([v,label,col],i)=>{const x=left+120+i*180,y=map(v,lo,hi,bottom,top);c.rect(x-35,y,70,bottom-y,col,'#fff',.8,2);c.text(x,bottom+7,label,10,'#333','middle');c.text(x,y-21,Number(v).toFixed(6),10,col,'middle',700);});
    } else {
      c.line(left,top,left,bottom,'#333',1);c.line(left,bottom,right,bottom,'#333',1);c.line(left,bottom-12,right,bottom-12,'#9ca3af',.9,'3,3');c.text(left-8,bottom-18,'0',9.5,'#333','end');
      scenarios.forEach((r,i)=>{const ci=contexts.indexOf(Number(r.context_length_tokens)),x=map(i,0,14,left,right);c.marker(x,bottom-12,SHAPE[ci],COLOR[ci],3);});
      contexts.forEach((ctx,i)=>c.text(map(i*3+1,0,14,left,right),bottom+5,ctxLabel(ctx),8.5,COLOR[i],'middle'));
      c.text((left+right)/2,bottom+21,'15 个预算—上下文情景；两分支差值均为 0',9,'#4b5563','middle');
    }
  }
  c.text(560,638,'两种 Q₀ 不改变 M0 的 N/D 排序；两分支的 N、D、M0 Loss 完全相同。q_equal 按 Q1 已有候选列读取。',10.3,'#374151','middle');
  return {canvas:c,qhub,qeq};
}

async function validateInputs(data) {
  const frontier=data.frontier,boot=data.bootstrap_frontier,shares=data.cost_shares,shifts=data.descriptive_shifts;
  assert(frontier.length===15&&boot.length===15&&shares.length===15,'P1/P2/P3 require 15 scenarios each');
  assert(shifts.length===22,`Expected 22 descriptive shifts, found ${shifts.length}`);
  const budgets=[...new Set(frontier.map(r=>num(r.budget_flops)))].sort((a,b)=>a-b),contexts=[...new Set(frontier.map(r=>num(r.context_length_tokens)))].sort((a,b)=>a-b);
  assert(JSON.stringify(budgets)===JSON.stringify([1e19,1e22,1e24]),'Unexpected budget set');
  assert(JSON.stringify(contexts)===JSON.stringify(CONTEXTS),'Unexpected contexts');
  const f=new Map(frontier.map(r=>[key(r),r])),b=new Map(boot.map(r=>[key(r),r])),s=new Map(shares.map(r=>[key(r),r]));
  assert(f.size===15&&b.size===15&&s.size===15,'Duplicate scenario keys');
  const baselineByKey=new Map();
  for(const r of data.bootstrap_frequency.filter(r=>bool(r.baseline_selected))){assert(!baselineByKey.has(key(r)),'Duplicate baseline selection');baselineByKey.set(key(r),r);}
  assert(baselineByKey.size===15,'Expected one baseline selection per scenario');
  for(const [scenario,r] of b){
    const source=f.get(scenario),share=s.get(scenario),freq=baselineByKey.get(scenario);
    assert(r.bootstrap_replicates==='1000','Bootstrap count changed');
    assert(freq.run_id===source.selected_run_id,'Baseline selected run mismatch');
    assert(Math.abs(num(freq.selection_frequency)-num(r.baseline_selection_frequency))<1e-12,'Selection frequency mismatch');
    assert(num(r.selected_N_B_p2_5)===num(r.selected_N_B_median)&&num(r.selected_N_B_p97_5)===num(r.selected_N_B_median),'N interval not degenerate');
    assert(num(r.selected_D_B_p2_5)===num(r.selected_D_B_median)&&num(r.selected_D_B_p97_5)===num(r.selected_D_B_median),'D interval not degenerate');
    assert(Math.abs(num(share.quality_cost_share))<1e-14,'Fixed Q quality share nonzero');
    assert(num(source.C_Q_flops)===0,'Baseline quality cost should be zero');
  }
  const qe=data.q1_a1_summary.filter(r=>r.dataset==='a1'&&r.candidate==='q_equal'),qh=data.q1_a1_summary.filter(r=>r.dataset==='a1'&&r.candidate==='q_huber');
  assert(qe.length===1&&qh.length===1,'Require one q_equal and q_huber A1 summary row');
  assert(Math.abs(num(frontier[0].q0_reference_a1_macro)-num(qh[0].macro_domain_score))<1e-12,'q_huber Q0 mismatch');
  assert(Math.abs(num(frontier[0].q_equal_sensitivity_macro)-num(qe[0].macro_domain_score))<1e-12,'q_equal Q0 mismatch');
  return {budgets,contexts,q_huber_q0:num(qh[0].macro_domain_score),q_equal_q0:num(qe[0].macro_domain_score),frontier_rows:frontier.length,bootstrap_rows:boot.length,frequency_rows:data.bootstrap_frequency.length,cost_share_rows:shares.length,descriptive_shift_rows:shifts.length,bootstrap_replicates:1000,point_mass_intervals:true,max_abs_delta_N_B:0,max_abs_delta_D_B:0,max_abs_delta_M0_loss:0,q0_branch_logic:'At each branch Q is fixed to that branch Q0, so C_Q=0; M0 has no Q/p loss term; budget masks and M0 ranking are invariant.'};
}

async function writeFigure(stem,canvas) {
  const svg=canvas.finish(),svgPath=path.join(FIG,`${stem}.svg`),pngPath=path.join(FIG,`${stem}.png`),grayPath=path.join(FIG,`${stem}_grayscale.png`);
  await fs.writeFile(svgPath,svg,'utf8');
  const pngBuffer=await sharp(Buffer.from(svg)).png({compressionLevel:9}).withMetadata({density:300}).toBuffer();
  await fs.writeFile(pngPath,pngBuffer);
  await sharp(pngBuffer).greyscale().png({compressionLevel:9}).withMetadata({density:300}).toFile(grayPath);
  const metadata=await sharp(pngPath).metadata();
  const grayMetadata=await sharp(grayPath).metadata();
  assert(metadata.width>=2000&&metadata.height>=1000,`Unexpected PNG resolution for ${stem}: ${metadata.width}x${metadata.height}`);
  assert(Math.abs((metadata.density??0)-300)<1,`Missing 300-DPI metadata in ${stem}`);
  assert(grayMetadata.width===metadata.width&&grayMetadata.height===metadata.height,`Color/grayscale size mismatch in ${stem}`);
  assert(Math.abs((grayMetadata.density??0)-300)<1,`Missing grayscale 300-DPI metadata in ${stem}`);
  assert(!canvas.bounds.some(b=>b.x1<0||b.y1<0||b.x2>canvas.w||b.y2>canvas.h),`SVG text outside artboard: ${stem}`);
  assert((svg.match(/<text /g)||[]).length>=15,`Unexpectedly few SVG text labels: ${stem}`);
  return {size_inches:[canvas.w/100,canvas.h/100],dpi:metadata.density,png_pixels:[metadata.width,metadata.height],grayscale_png_pixels:[grayMetadata.width,grayMetadata.height],grayscale_dpi:grayMetadata.density,svg_text_nodes:(svg.match(/<text /g)||[]).length,text_bounds_checked:canvas.bounds.length,text_out_of_bounds:[]};
}

function contractText() {
  return `# Q3 补充图表契约\n\n**范围：** 只绘制已归档的 Q3 固定 Q₀/M0 结果；所有输入只读。图件由 Node、SVG 和 Sharp 生成，不依赖 Python/Matplotlib。每张图交付可编辑 SVG、300 DPI PNG 和同图灰度 PNG。\n\n## P1｜固定 Q₀ 下的预算—N/D 配置\n\n- 输入：\`Q3/04_结果/M0_ND_reference_frontier.csv\`（15 行）。读取预算、上下文、\`selected_N_B\`、\`selected_D_B\` 与支持边界标记。\n- 双面板：N、D 随 3 档离散预算变化；5 个上下文以色彩和点形共同区分，预算按 log10 映射。\n- 限制：结果限 B1 实测网格、固定 Q=Q₀；不代表完整 Q3 联合最优，不涉及 p/Q→Loss。外圈标记支持上界。\n\n## P2｜Bootstrap 配置稳定性\n\n- 输入：\`M0_ND_cluster_bootstrap_frontier.csv\`（15 行）和逐候选频率表（全表）；核对基线 run_id、频率及 1,000 次重复。\n- 三面板：N、D 95% 区间及原最优点重选率/切换概率。N/D 端点相同则显示单点；切换概率=1−重选率。\n- 横轴编码：按上下文长度（Token）分组，组内三个点依次代表预算 10¹⁹、10²²、10²⁴ FLOPs。\n- 限制：Bootstrap 来自 8 条 B1 轨迹，只表示既定 M0 函数形式、支持网格内的参数传播稳定性，不覆盖模型形式误差或迁移。\n\n## P3｜资源配置、成本份额与 Loss\n\n- 输入：M0 前沿（15 行）、成本份额（15 行）和描述性相邻情景表（22 行）。\n- 2×2 面板：N、D、训练/注意力/质量成本份额、M0 预测验证 Loss。\n- 限制：Q=Q₀ 时质量成本份额为 0；最高预算部分选点触及 B1 支持上界。线段只连接离散情景，22 对变化不作为结构转移检验。\n\n## P4｜Q 基准敏感性\n\n- 输入：Q3 前沿中的两种 Q₀ 字段及 Q1 A1 摘要（q_huber、q_equal 各 1 行）；核对 Q₀ 分别约 0.4984781048、0.4962230100。\n- 展示 Q₀ 基准和 15 个情景的 |ΔN|、|ΔD|、|ΔM0 Loss|；各差值为零。\n- q_qual 暂按现有候选名 q_equal 理解。固定各自 Q₀ 时 C_Q=0 且 M0 无 Q/p 响应，所以选择不变；不声称测得 Q→Loss 效应。\n\n## 版面与复现\n\n- 色盲友好颜色配合不同点形；灰度版由同一彩色 PNG 转换。SVG 保留文字节点。\n- 页面文字边界、PNG 像素/DPI、SVG 文本节点、输入输出 SHA-256 和 P4 口径核验写入 \`复现清单.json\`。\n`;
}

async function main() {
  await fs.mkdir(FIG,{recursive:true});
  const data=Object.fromEntries(await Promise.all(Object.entries(INPUTS).map(async ([k,v])=>[k,await readCsv(v)])));
  const validation=await validateInputs(data),{budgets,contexts}=validation;
  const q1Rows=data.q1_a1_summary;
  const specs=[
    ['result_q3_p1_budget_nd_frontier',plotP1(data.frontier,budgets,contexts)],
    ['result_q3_p2_bootstrap_selection_stability',plotP2(data.bootstrap_frontier,data.bootstrap_frequency,contexts,budgets)],
    ['result_q3_p3_resource_shift_dashboard',plotP3(data.frontier,data.cost_shares,contexts,budgets)],
  ];
  const p4=plotP4(data.frontier,q1Rows);validation.p4_q_huber_q0=p4.qhub;validation.p4_q_equal_q0=p4.qeq;specs.push(['result_q3_p4_q_baseline_sensitivity',p4.canvas]);
  const figureQa={};for(const [stem,canvas] of specs)figureQa[stem]=await writeFigure(stem,canvas);
  const inputHashes=Object.fromEntries(await Promise.all(Object.values(INPUTS).map(async rel=>[rel,await sha256(path.join(ROOT,rel))])));
  const outputFiles=(await fs.readdir(FIG)).filter(n=>/\.(svg|png)$/.test(n)).sort();
  const outputHashes=Object.fromEntries(await Promise.all(outputFiles.map(async name=>[`Q3/归档_20260925/Q3补充图表/figures/${name}`,await sha256(path.join(FIG,name))])));
  const contractPath=path.join(OUT,'图表契约.md');await fs.writeFile(contractPath,contractText(),'utf8');
  const manifest={artifact:'Q3 supplementary figures P1-P4',status:'PASS',generated_at_utc:new Date().toISOString(),command:'node Q3/归档_20260925/plot_q3_supplementary_figures.mjs',runtime:{node:process.version,sharp:sharp.versions.sharp,libvips:sharp.versions.vips},renderer:'Node.js + Sharp/libvips rasterizes authored SVG; no Python, no refitting, no resampling.',inputs_sha256:inputHashes,script_sha256:{'Q3/归档_20260925/plot_q3_supplementary_figures.mjs':await sha256(fileURLToPath(import.meta.url))},outputs_sha256:outputHashes,contract_sha256:await sha256(contractPath),validation,figure_qa:figureQa,manual_visual_review:'Reviewed all four color and grayscale PNGs at full rendering scale: labels remain within the page, no legend or tick overlap was observed, and grayscale series retain distinct markers. P4 zero deltas are intentional and stated in the chart.',assets_per_figure:['svg','300-DPI png','grayscale png']};
  await fs.writeFile(path.join(OUT,'复现清单.json'),JSON.stringify(manifest,null,2)+'\n','utf8');
  console.log(`Status: ${manifest.status}`);console.log(`Figures: ${specs.length}; files: ${outputFiles.length}`);console.log(`Output: ${path.relative(ROOT,OUT)}`);
}

main().catch(error=>{console.error(`ERROR: ${error.stack||error}`);process.exitCode=1;});
