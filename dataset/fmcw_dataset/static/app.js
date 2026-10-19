'use strict';
const $ = id => document.getElementById(id);
const token = document.querySelector('meta[name=local-token]').content;
let config, initial, currentDomain, validationTimer, revision=0, running=false, pollTimer;
let presets=[];
let plotScale='linear';
let rangeFields=[]; let previewData=null, previewConfig=null, plotBounds=null, folderState=null, previewRevision=0;
const fmt=(v,n=3)=>Number(v).toLocaleString('en-US',{maximumFractionDigits:n});
const get=path=>path.split('.').reduce((o,k)=>o[k],config);
const set=(path,v)=>{const p=path.split('.'),k=p.pop();p.reduce((o,k)=>o[k],config)[k]=v;};
const clone=v=>JSON.parse(JSON.stringify(v));
const sorted=v=>Array.isArray(v)?v.map(sorted):v&&typeof v==='object'?Object.fromEntries(Object.keys(v).sort().map(k=>[k,sorted(v[k])])):v;
const escapeText=v=>String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function status(message,type=''){ $('status').textContent=message;$('status').className=type; }
async function api(path,data){
  const options={headers:{'X-Local-Token':token}};
  if(data!==undefined){options.method='POST';options.headers['Content-Type']='application/json';options.body=JSON.stringify(data);}
  const r=await fetch('/api/'+path,options); const body=await r.json();
  if(!r.ok)throw Error(body.error||'Local server error.');return body;
}
function rangeField(parent,path,label,lo,hi,step=0.01){
  const row=document.createElement('div');row.className='range-row';row.dataset.range=path;
  const id='field-'+path.replaceAll('.','-');
  row.innerHTML=`<label for="${id}-low">${label}</label><div class="range-track"><div class="range-fill"></div><button type="button" class="range-handle" role="slider" aria-label="${label} minimum"></button><button type="button" class="range-handle" role="slider" aria-label="${label} maximum"></button></div><input class="low" id="${id}-low" type="number" step="any" aria-label="${label} minimum"><span class="dash">–</span><input class="high" id="${id}-high" type="number" step="any" aria-label="${label} maximum">`;
  $(parent).append(row);
  const inputs=[...row.querySelectorAll('input')],handles=[...row.querySelectorAll('.range-handle')],track=row.querySelector('.range-track'),fill=row.querySelector('.range-fill');
  const field={row,path,lo,hi,step,refresh(){
    const pair=get(path);let min=this.lo,max=this.hi;
    if(path!=='targets.distance_m'){min=Math.min(min,...pair);max=Math.max(max,...pair);}
    if(!(max>min))max=min+1;
    row.classList.toggle('invalid',pair[0]>pair[1]||(path==='targets.distance_m'&&(pair[0]<this.lo||pair[1]>this.hi)));
    inputs.forEach((input,i)=>{if(document.activeElement!==input)input.value=pair[i];if(path==='targets.distance_m'){input.min=Math.max(this.lo,0.000001);input.max=this.hi;}});
    const pos=pair.map(v=>Math.max(0,Math.min(100,100*(v-min)/(max-min))));
    handles.forEach((h,i)=>{h.style.left=pos[i]+'%';h.setAttribute('aria-valuemin',min);h.setAttribute('aria-valuemax',max);h.setAttribute('aria-valuenow',pair[i]);h.title=String(pair[i]);});
    fill.style.left=pos[0]+'%';fill.style.width=Math.max(0,pos[1]-pos[0])+'%';
    this.scale=[min,max];
  }};
  inputs.forEach((input,i)=>input.addEventListener('input',()=>{get(path)[i]=input.valueAsNumber;changed();}));
  function move(event,i){const rect=track.getBoundingClientRect(),[min,max]=field.scale;const fraction=Math.max(0,Math.min(1,(event.clientX-rect.left)/rect.width));let v=min+fraction*(max-min);v=Number((Math.round(v/field.step)*field.step).toPrecision(12));v=Math.max(min,Math.min(max,v));const p=get(path);p[i]=i===0?Math.min(v,p[1]):Math.max(v,p[0]);changed();}
  handles.forEach((handle,i)=>{
    handle.addEventListener('pointerdown',event=>{if(running)return;event.preventDefault();handle.setPointerCapture(event.pointerId);move(event,i);});
    handle.addEventListener('pointermove',event=>{if(handle.hasPointerCapture(event.pointerId))move(event,i);});
    handle.addEventListener('keydown',event=>{let delta=event.key==='ArrowRight'||event.key==='ArrowUp'?step:event.key==='ArrowLeft'||event.key==='ArrowDown'?-step:0;if(!delta)return;event.preventDefault();const p=get(path);let v=Math.max(field.scale[0],Math.min(field.scale[1],p[i]+delta*(event.shiftKey?10:1)));p[i]=Number((i===0?Math.min(v,p[1]):Math.max(v,p[0])).toPrecision(12));changed();});
  });
  rangeFields.push(field);field.refresh();return field;
}
function render(){
  rangeFields=[];
  for(const id of ['radar-fields','target-fields','rcs-range','noise-fields','neighbor-fields','neighbor-levels','neighbor-advanced','profile-fields'])$(id).replaceChildren();
  rangeField('radar-fields','radar.duration_us','Duration (µs)',0.1,100,0.1);
  rangeField('radar-fields','radar.bandwidth_mhz','Bandwidth (MHz)',1,4000,1);
  rangeField('radar-fields','radar.sample_rate_mhz','Sample rate (MHz)',0.1,200,0.1);
  rangeField('radar-fields','radar.frequency_ghz','Frequency (GHz)',76,81,0.01);
  rangeField('target-fields','targets.count','Target count',1,20,1);
  rangeField('target-fields','targets.distance_m','Distance (m)',0,74.648322,0.01);
  rangeField('rcs-range','targets.rcs_m2','RCS (m²)',0.001,100,0.1);
  rangeField('noise-fields','noise.rms','Complex RMS',0,0.01,0.00001);
  rangeField('noise-fields','noise.snr_db','Time-domain SNR (dB)',-60,60,0.1);
  rangeField('neighbor-fields','neighbors.count','Interferer count',1,10,1);
  rangeField('neighbor-fields','neighbors.slope_factor','Slope factor k',0.01,3,0.01);
  rangeField('neighbor-levels','neighbors.level','Amplitude per interferer',0,1,0.001);
  rangeField('neighbor-levels','neighbors.sir_db','Total SIR (dB)',-60,60,0.1);
  rangeField('neighbor-advanced','neighbors.bandwidth_mhz','Bandwidth (MHz)',1,4000,1);
  rangeField('neighbor-advanced','neighbors.duration_us','Duration (µs)',0.1,100,0.1);
  rangeField('neighbor-advanced','neighbors.frequency_offset_mhz','Frequency offset (MHz)',-1000,1000,1);
  rangeField('neighbor-advanced','neighbors.offset_us','Time offset (µs)',-100,100,0.1);
  ['Targets + interference','Targets only','Interference only','Noise only'].forEach((text,i)=>{
    const label=document.createElement('label');label.innerHTML=`${text}<span><input type="number" min="0" max="100" step="any" aria-label="Proportion ${text}" value="${config.profile[i]}"> %</span>`;
    label.querySelector('input').oninput=e=>{config.profile[i]=e.target.valueAsNumber;changed();};$('profile-fields').append(label);
  });
  document.querySelectorAll('[data-path]').forEach(el=>el.value=get(el.dataset.path));
  syncVisibility();changed();
}
function syncVisibility(){
  $('rcs-fixed-row').hidden=config.targets.rcs_mode!=='fixed';$('rcs-range').hidden=config.targets.rcs_mode==='fixed';
  document.querySelectorAll('[data-rcs]').forEach(b=>{b.classList.toggle('active',b.dataset.rcs===config.targets.rcs_mode);b.setAttribute('aria-pressed',b.dataset.rcs===config.targets.rcs_mode);});
  rangeFields.find(x=>x.path==='noise.snr_db').row.hidden=config.noise.mode!=='snr';
  rangeFields.find(x=>x.path==='neighbors.sir_db').row.hidden=config.neighbors.power_mode!=='sir';
  rangeFields.find(x=>x.path==='neighbors.slope_factor').row.hidden=config.neighbors.mode!=='linked';
  ['neighbors.bandwidth_mhz','neighbors.duration_us'].forEach(p=>rangeFields.find(x=>x.path===p).row.hidden=config.neighbors.mode!=='independent');
}
function changed(){
  revision++;const rev=revision;
  if(!running){$('generate').disabled=true;$('preview').disabled=true;status('Checking…');}
  document.querySelector('.suffix').textContent=config.name.endsWith('.h5')?'':'.h5';
  rangeFields.forEach(f=>f.refresh());syncVisibility();
  const signature=c=>{const v=clone(c);delete v.folder;delete v.name;return JSON.stringify(sorted(v));};
  $('preset').value=presets.find(p=>signature(p.config)===signature(config))?.id||'custom';
  clearTimeout(validationTimer);
  validationTimer=setTimeout(()=>validateForm(rev),220);
}
async function validateForm(rev){
  try{
    const meta=await api('domain',{radar:config.radar});if(rev!==revision)return;
    currentDomain=meta.domain;const r=meta.radar,d=meta.domain;
    const dist=rangeFields.find(x=>x.path==='targets.distance_m');dist.lo=Math.max(0.000001,d.min_distance_m);dist.hi=d.max_distance_m;dist.refresh();
    const c=config.radar;
    const interval=(lo,hi,n=3)=>lo===hi?fmt(lo,n):fmt(lo,n)+'–'+fmt(hi,n);
    const resolution=c.bandwidth_mhz.map(b=>299792458/(2*b*1e6)).reverse();
    $('radar-summary').textContent=`Range ${interval(d.min_distance_m,d.max_distance_m)} m · ΔR ${interval(...resolution)} m · N ${interval(r.N,d.max_points,0)}`;
    $('radar-summary').title='Common range across configurations, with a one-bin margin. ΔR = c/(2B).';
    const validated=await api('validate',config);if(rev!==revision)return;
    $('estimate').textContent=`≤ ${fmt(validated.signal_bytes_max/1e6,1)} MB of signals`;
    if(!running){$('generate').disabled=false;$('preview').disabled=false;status('Ready.');}
  }catch(e){if(rev===revision&&!running){$('generate').disabled=true;$('preview').disabled=true;status(e.message,'error');}}
}
document.querySelectorAll('[data-path]').forEach(el=>{
  el.addEventListener('change',()=>{
    const path=el.dataset.path;let value=el.type==='number'?el.valueAsNumber:el.value;
    if(path==='fft_factor')value=Number(value);
    if(path==='radar.frequency_reference'){
      const B=config.radar.bandwidth_mhz;
      if(B[0]!==B[1]){el.value=config.radar.frequency_reference;status('Set a fixed bandwidth before changing the frequency reference.','error');return;}
      const k={start:0,center:.5,end:1},delta=(k[value]-k[config.radar.frequency_reference])*B[0]/1000;
      config.radar.frequency_ghz=config.radar.frequency_ghz.map(f=>Number((f+delta).toPrecision(12)));
    }
    set(path,value);changed();
  });
});
document.querySelectorAll('[data-rcs]').forEach(b=>b.onclick=()=>{config.targets.rcs_mode=b.dataset.rcs;changed();});
document.querySelectorAll('[data-close]').forEach(b=>b.onclick=()=>$(b.dataset.close).close());
$('random-seed').onclick=()=>{config.seed=crypto.getRandomValues(new Uint32Array(1))[0];$('seed').value=config.seed;changed();};
$('preset').onchange=()=>{
  const preset=presets.find(p=>p.id===$('preset').value);
  if(!preset)return;
  const folder=config.folder;
  config=clone(preset.config);config.folder=folder;initial=clone(config);render();
};
$('reset').onclick=()=>{config=clone(initial);render();};
$('save-config').onclick=()=>{const blob=new Blob([JSON.stringify(config,null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=(config.name||'settings')+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
$('load-config').onclick=()=>$('config-file').click();
$('config-file').onchange=async e=>{try{const next=JSON.parse(await e.target.files[0].text());await api('validate',next);config=next;render();}catch(err){status('Settings not loaded: '+err.message,'error');}e.target.value='';};
async function folders(path){
  try{folderState=await api('folders',{path});$('folder-path').value=folderState.path;$('folder-list').replaceChildren();for(const entry of folderState.entries){const b=document.createElement('button');b.textContent='▱  '+entry.name;b.onclick=()=>folders(entry.path);$('folder-list').append(b);}}catch(e){status(e.message,'error');}
}
$('browse').onclick=async()=>{await folders(config.folder||'');if(!folderState)await folders('');$('folder-dialog').showModal();};
$('parent-folder').onclick=()=>folders(folderState.parent);
$('go-folder').onclick=()=>folders($('folder-path').value);
$('folder-path').onkeydown=e=>{if(e.key==='Enter')folders(e.target.value);};
$('choose-folder').onclick=()=>{config.folder=folderState.path;$('folder').value=config.folder;$('folder-dialog').close();changed();};
function lockForm(lock){
  running=lock;document.querySelectorAll('#form-scroll input,#form-scroll select,#form-scroll button,#reset,#generate,#preview').forEach(el=>el.disabled=lock);
  $('cancel').hidden=!lock;$('cancel').disabled=false;
}
$('generate').onclick=async()=>{
  try{await api('validate',config);const job=await api('generate',config);lockForm(true);$('open-output').hidden=true;status('Generating…');showJob(job);pollTimer=setInterval(poll,350);}catch(e){status(e.message,'error');}
};
function showJob(job){
  const pct=job.total?Math.round(100*job.done/job.total):0;$('progress').value=pct;$('progress-text').textContent=pct+' %';
  $('stat-count').textContent=`${job.done} / ${job.total} scenes`;$('stat-speed').textContent=`${fmt(job.speed||0,1)} scenes/s`;
  $('stat-time').textContent=`${fmt(job.elapsed||0,1)} s`;$('stat-eta').textContent=`Remaining: ${job.eta==null?'—':fmt(job.eta,1)+' s'}`;
}
async function poll(){try{const job=await api('status');showJob(job);if(job.status==='running')return;clearInterval(pollTimer);lockForm(false);if(job.status==='complete'){
    const r=job.result;$('stat-size').textContent=`Size: ${fmt(r.bytes/1e6,2)} MB`;$('open-output').hidden=false;
    status(`Saved: ${r.path}`, 'success');
  }else status(job.message||job.status,job.status==='error'?'error':'');
}catch(e){status('Connection lost: '+e.message,'error');}}
$('cancel').onclick=async()=>{try{await api('cancel',{});$('cancel').disabled=true;status('Cancelling…');}catch(e){status(e.message,'error');}};
$('open-output').onclick=()=>api('open-folder',{}).catch(e=>status(e.message,'error'));
$('quit').onclick=async()=>{if(running){status('Cancel generation before closing.','error');return;}try{await api('quit',{});clearInterval(pollTimer);clearTimeout(validationTimer);document.querySelectorAll('button,input,select').forEach(el=>el.disabled=true);status('Stopped. You can close this page.');}catch(e){status(e.message,'error');}};
async function loadPreview(){
  const requestRevision=++previewRevision;
  $('preview-message').textContent='Loading preview…';
  try{const next=await api('preview',{config:previewConfig,index:Number($('preview-index').value),window:$('plot-window').value,cfar:{enabled:$('cfar-enabled').checked,training:Number($('cfar-training').value),guard:Number($('cfar-guard').value),pfa:Number($('cfar-pfa').value)}});if(requestRevision!==previewRevision)return;previewData=next;plotBounds=null;drawPlots();const r=previewData.radar,i=previewData.info;
    const ratio=x=>x===null?'no targets':typeof x==='string'?x+' dB':fmt(x,2)+' dB';
    $('preview-summary').textContent=`Sample ${previewData.index} · ${r.N} points · ${previewData.targets.length} target(s), ${previewData.neighbors.length} interferer(s) · SNR ${ratio(i.snr_db)} · SIR ${ratio(i.sir_db)} · active fraction ${fmt(i.active_fraction*100,1)} %`;
    $('target-list').textContent='Targets: '+(previewData.targets.map(t=>`${fmt(t.distance,3)} m · RCS ${fmt(t.rcs,2)} m²`).join(' | ')||'none');
    const notes=[];if(previewData.neighbors.length&&i.pi===0)notes.push('No received interference.');if(previewConfig.noise.mode==='snr'&&!i.snr_applied)notes.push('No targets: using RMS.');if(previewConfig.neighbors.power_mode==='sir'&&!i.sir_applied)notes.push('SIR setting not applicable.');
    $('preview-message').textContent=notes.join(' ');
  }catch(e){if(requestRevision===previewRevision)$('preview-message').textContent=e.message;}
}
$('preview').onclick=async()=>{try{await api('validate',config);previewConfig=clone(config);$('preview-dialog').showModal();await loadPreview();}catch(e){status(e.message,'error');}};
$('preview-next').onclick=()=>{$('preview-index').value=Number($('preview-index').value)+1;loadPreview();};
$('preview-index').onchange=loadPreview;$('plot-window').onchange=loadPreview;
for(const id of ['cfar-enabled','cfar-training','cfar-guard','cfar-pfa'])$(id).onchange=loadPreview;
document.querySelectorAll('[data-scale]').forEach(button=>button.onclick=()=>{
  plotScale=button.dataset.scale;
  document.querySelectorAll('[data-scale]').forEach(b=>{const active=b.dataset.scale===plotScale;b.classList.toggle('active',active);b.setAttribute('aria-pressed',String(active));});
  drawPlots();
});
$('plot-view').onchange=()=>{plotBounds=null;drawPlots();};$('zoom-reset').onclick=()=>{plotBounds=null;drawPlots();};
document.querySelectorAll('[data-wave]').forEach(el=>el.onchange=drawPlots);
const colors={s:'#21915d',sn:'#66788b',y:'#FA3239'};
const amplitudeValue=value=>plotScale==='linear'?value:20*Math.log10(Math.max(1e-15,value));
const chartNumber=value=>value!==0&&Math.abs(value)<.01?value.toExponential(2):fmt(value,4);
function chartData(){
  const d=previewData,view=$('plot-view').value,series=[];
  const x=view==='time'?d.time_us:view==='frequency'?d.frequency_mhz:d.distance_m;
  for(const key of ['s','sn','y'])if(document.querySelector(`[data-wave="${key}"]`).checked){
    if(view==='time'){
      series.push({key:key+' I',color:colors[key],values:d.temporal[key].real});
      series.push({key:key+' Q',color:colors[key],values:d.temporal[key].imag,dash:true});
    }else{const z=d.spectrum[key];const linear=plotScale==='linear';series.push({key,color:colors[key],values:z.real.map((re,i)=>linear?Math.hypot(re,z.imag[i]):20*Math.log10(Math.max(1e-15,Math.hypot(re,z.imag[i]))))});}
  }
  if(view!=='time'&&d.cfar?.settings.enabled&&$('cfar-enabled').checked){
    for(const key of ['s','y'])if(document.querySelector(`[data-wave="${key}"]`).checked&&d.cfar.signals[key]){
      series.push({key:'CFAR threshold '+key,color:colors[key],dash:true,values:d.cfar.signals[key].threshold_power.map(p=>p===null?null:amplitudeValue(Math.sqrt(Math.max(0,p))))});
    }
  }
  return {x,series,xlabel:view==='time'?'Time (µs)':view==='frequency'?'Signed FFT frequency (MHz)':'Distance (m)',ylabel:view==='time'?'Relative amplitude':plotScale==='linear'?'FFT amplitude (linear)':'FFT amplitude (dB, reference 1)'};
}
let chartTransform=null;
function drawChart(canvas,x,series,limits,xlabel,ylabel,forcedY=null,targets=[]){
  const rect=canvas.getBoundingClientRect();if(!rect.width)return;
  const dpr=devicePixelRatio||1;canvas.width=Math.round(rect.width*dpr);canvas.height=Math.round(rect.height*dpr);
  const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);const w=rect.width,h=rect.height,p={l:65,r:18,t:26,b:36},pw=w-p.l-p.r,ph=h-p.t-p.b;
  let ymin=Infinity,ymax=-Infinity;
  for(const s of series)for(let i=0;i<x.length;i++)if(x[i]>=limits[0]&&x[i]<=limits[1]&&Number.isFinite(s.values[i])){ymin=Math.min(ymin,s.values[i]);ymax=Math.max(ymax,s.values[i]);}
  if(forcedY)[ymin,ymax]=forcedY;else if(!Number.isFinite(ymin)||ymax-ymin<1e-12){ymin=-1;ymax=1;}else{let pad=(ymax-ymin)*.08;ymin-=pad;ymax+=pad;}
  if(canvas.id==='signal-chart'&&$('plot-view').value!=='time'&&plotScale==='linear')ymin=0;
  const xp=v=>p.l+(v-limits[0])/(limits[1]-limits[0])*pw,yp=v=>p.t+(ymax-v)/(ymax-ymin)*ph;
  ctx.fillStyle='#fff';ctx.fillRect(0,0,w,h);ctx.font='11px Segoe UI,Arial';ctx.lineWidth=1;
  for(let k=0;k<=5;k++){const y=p.t+k*ph/5;ctx.strokeStyle='#e7eaf0';ctx.beginPath();ctx.moveTo(p.l,y);ctx.lineTo(w-p.r,y);ctx.stroke();ctx.fillStyle='#647083';ctx.textAlign='right';ctx.fillText(chartNumber(ymax-k*(ymax-ymin)/5),p.l-9,y+4);}
  for(let k=0;k<=6;k++){const xv=limits[0]+k*(limits[1]-limits[0])/6,px=xp(xv);ctx.strokeStyle='#e7eaf0';ctx.beginPath();ctx.moveTo(px,p.t);ctx.lineTo(px,h-p.b);ctx.stroke();ctx.textAlign='center';ctx.fillStyle='#647083';ctx.fillText(fmt(xv,2),px,h-p.b+17);}
  ctx.textAlign='left';ctx.fillStyle='#465267';ctx.fillText(ylabel,p.l,15);ctx.textAlign='right';ctx.fillText(xlabel,w-p.r,h-5);
  ctx.save();ctx.beginPath();ctx.rect(p.l,p.t,pw,ph);ctx.clip();
  for(const s of series){ctx.strokeStyle=s.color;ctx.lineWidth=1.1;ctx.setLineDash(s.dash?[4,3]:[]);ctx.beginPath();let first=true;for(let i=0;i<x.length;i++){if(x[i]<limits[0]||x[i]>limits[1]||!Number.isFinite(s.values[i])){first=true;continue;}const px=xp(x[i]),py=yp(s.values[i]);if(first){ctx.moveTo(px,py);first=false;}else ctx.lineTo(px,py);}ctx.stroke();}
  ctx.setLineDash([3,4]);for(const t of targets){if(t<limits[0]||t>limits[1])continue;ctx.strokeStyle='#21915d88';ctx.beginPath();ctx.moveTo(xp(t),p.t);ctx.lineTo(xp(t),p.t+ph);ctx.stroke();}ctx.restore();
  return {limits,p,pw,ph,ymin,ymax,xp,yp,w,h};
}
function drawPlots(){
  if(!previewData||!$('preview-dialog').open)return;
  const data=chartData(),view=$('plot-view').value;
  if(!plotBounds){plotBounds=view==='distance'?[Math.max(0,previewData.radar.min_distance_m),previewData.radar.max_distance_m]:[data.x[0],data.x.at(-1)];}
  const targetXs=view==='time'?[]:previewData.targets.map(t=>view==='distance'?t.distance:(2*previewData.radar.slope*t.distance/299792458-previewData.radar.shift)/1e6);
  chartTransform=drawChart($('signal-chart'),data.x,data.series,plotBounds,data.xlabel,data.ylabel,null,targetXs);
  drawCfar(data,view);
  $('label-chart').parentElement.hidden=view==='time';
  if(view!=='time')drawChart($('label-chart'),data.x,[{values:previewData.label,color:'#21915d'}],plotBounds,data.xlabel,'Target presence (label)',[0,1.1],targetXs);
}

function drawCfar(data,view){
  $('cfar-panel').hidden=view==='time';$('cfar-results').hidden=view==='time';
  if(view==='time')return;
  const results=previewData.cfar;
  if(!results?.settings.enabled||!$('cfar-enabled').checked){$('cfar-results').textContent='CFAR disabled.';return;}
  $('cfar-results').replaceChildren();
  const assessment=previewData.detection_assessment;
  if(assessment){
    const row=document.createElement('p');
    const before=assessment.baseline,after=assessment.observed,lost=assessment.lost_distances_m;
    row.textContent=`Detection before/after: ${before.tp}/${assessment.target_count} → ${after.tp}/${assessment.target_count} · Lost: ${lost.length} · False peaks: ${before.fp} → ${after.fp}`;
    $('cfar-results').append(row);
  }

  for(const key of ['s','y']){
    const detected=results.signals[key];if(!detected)continue;
    const row=document.createElement('p');row.className=key==='s'?'clean-text':'interfered-text';
    row.textContent=(key==='s'?'Green · s':'Red · y')+' : '+detected.indices.length+' peak(s)'+(detected.distance_m.length?' — '+detected.distance_m.map(v=>fmt(v,3)+' m').join(' · '):'');
    $('cfar-results').append(row);
    if(!document.querySelector(`[data-wave="${key}"]`).checked||!chartTransform)continue;
    const t=chartTransform,ctx=$('signal-chart').getContext('2d');
    ctx.save();ctx.beginPath();ctx.rect(t.p.l,t.p.t,t.pw,t.ph);ctx.clip();ctx.strokeStyle=colors[key];ctx.lineWidth=1.6;ctx.setLineDash([]);
    detected.indices.forEach((index,i)=>{
      const x=data.x[index];if(x<plotBounds[0]||x>plotBounds[1])return;
      const px=t.xp(x),py=t.yp(amplitudeValue(detected.amplitude[i]));ctx.beginPath();
      if(key==='s')ctx.arc(px,py,5.5,0,2*Math.PI);
      else{ctx.moveTo(px,py-5);ctx.lineTo(px-4.5,py+4);ctx.lineTo(px+4.5,py+4);ctx.closePath();}
      ctx.stroke();
    });ctx.restore();
  }
}

const canvas=$('signal-chart');let dragging=null;
canvas.addEventListener('wheel',e=>{if(!chartTransform)return;e.preventDefault();const t=chartTransform,rect=canvas.getBoundingClientRect(),f=Math.max(0,Math.min(1,(e.clientX-rect.left-t.p.l)/t.pw)),span=plotBounds[1]-plotBounds[0],factor=e.deltaY<0?.8:1.25,center=plotBounds[0]+f*span;plotBounds=[center-f*span*factor,center+(1-f)*span*factor];drawPlots();},{passive:false});
canvas.onpointerdown=e=>{if(!plotBounds)return;dragging={x:e.clientX,bounds:[...plotBounds]};canvas.setPointerCapture(e.pointerId);};
canvas.onpointerup=()=>{dragging=null;};canvas.ondblclick=()=>{plotBounds=null;drawPlots();};
canvas.onpointermove=e=>{
  if(!chartTransform||!previewData)return;const t=chartTransform,rect=canvas.getBoundingClientRect();
  if(dragging){const dx=(e.clientX-dragging.x)/t.pw*(dragging.bounds[1]-dragging.bounds[0]);plotBounds=dragging.bounds.map(v=>v-dx);drawPlots();return;}
  const d=chartData(),v=plotBounds[0]+(e.clientX-rect.left-t.p.l)/t.pw*(plotBounds[1]-plotBounds[0]);
  let lo=0,hi=d.x.length-1;while(lo<hi){const m=(lo+hi)>>1;if(d.x[m]<v)lo=m+1;else hi=m;}
  if(lo>0&&Math.abs(d.x[lo-1]-v)<Math.abs(d.x[lo]-v))lo--;
  const tip=$('chart-tooltip');tip.textContent=fmt(d.x[lo],6)+' '+(d.xlabel.includes('Time')?'µs':d.xlabel.includes('MHz')?'MHz':'m')+'\n'+d.series.filter(s=>Number.isFinite(s.values[lo])).map(s=>s.key+' : '+chartNumber(s.values[lo])).join('\n');tip.style.display='block';tip.style.left=Math.max(0,Math.min(rect.width-180,e.clientX-rect.left+12))+'px';tip.style.top='30px';
};canvas.onpointerleave=()=>{$('chart-tooltip').style.display='none';};
window.addEventListener('resize',drawPlots);
(async()=>{try{
  const r=await api('defaults',{});presets=r.presets||[];
  const labels={a_standard:'A · Standard',b_long:'B · Long ramp',c_large_bande:'C · Wide bandwidth',d_court:'D · Short ramp',e_unilateral:'E · Unilateral',a_support_connu:'A · Nominal',b_combinaison_exclue:'B · New combination',c_extrapolation_bande:'C · Extrapolation',d_combinaison_unilaterale:'D · Unilateral'};
  for(const group of ['principal','external']){
    const node=document.createElement('optgroup');node.label=group==='principal'?'Main · 24,000 scenes':'External · 6,000 scenes';
    for(const preset of presets.filter(p=>p.group===group)){
      const option=document.createElement('option');option.value=preset.id;option.textContent=(labels[preset.config.name]||preset.config.name)+' · '+fmt(preset.config.count,0);node.append(option);
    }
    if(node.children.length)$('preset').append(node);
  }
  config=clone(presets[0]?.config||r.config);config.folder=r.config.folder;initial=clone(config);render();
  const job=await api('status');if(job.status==='running'){lockForm(true);pollTimer=setInterval(poll,350);}
}catch(e){status('Cannot start: '+e.message,'error');}})();
