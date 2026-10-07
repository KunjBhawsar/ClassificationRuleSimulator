const $ = x => document.getElementById(x);
let ds = null, cols = [], lastImportance = [], lastRules = [], comparisonLoaded = false, modelReady = false;
const titles = {
  overview:['Overview','Classification rule mining dashboard'],
  dataset:['Dataset & Preprocessing','Inspect and prepare your CSV'],
  train:['Model Training','Train and tune a supervised classifier'],
  rules:['Classification Rules','Readable IF–THEN rules with confidence and support'],
  evaluation:['Evaluation','Macro metrics and confusion matrix'],
  visualization:['Visualization','Dataset-driven histograms and bar graphs'],
  prediction:['Prediction','Classify a new record']
};

document.querySelectorAll('nav button').forEach(b => b.onclick = () => show(b.dataset.p));
function show(p){
  document.querySelectorAll('.page').forEach(x=>x.classList.remove('active'));
  $(p).classList.add('active');
  document.querySelectorAll('nav button').forEach(x=>x.classList.toggle('active',x.dataset.p===p));
  $('title').textContent=titles[p][0]; $('sub').textContent=titles[p][1];
}
function esc(x){return String(x??'').replace(/[&<>"']/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[m]))}
function pct(x){return (Number(x)*100).toFixed(1)+'%'}
function num(x){return Number(x||0).toLocaleString()}

function renderPreprocessing(d){
  const p=d.preprocessing||{};
  const missingTotal=Number(p.missing_total??d.missing_values??0);
  $('missingPrep').innerHTML=`<b>${num(missingTotal)}</b> missing values${missingTotal?' → numeric values are median-imputed and categorical values use the most frequent value.':' → none detected.'}`;
  $('duplicatePrep').innerHTML=`<b>${num(p.duplicates??d.duplicate_rows??0)}</b> duplicate rows detected.`;
  const cats=p.categorical_columns||[];
  $('encodingPrep').innerHTML=`<b>${cats.length}</b> categorical columns → one-hot encoded during training.<br><small>${esc(cats.slice(0,8).join(', ')||'None')}</small>`;
  const ids=p.identifier_columns||d.ignored_id_columns||[];
  $('idPrep').innerHTML=ids.length?`Removed from model: <b>${esc(ids.join(', '))}</b>`:'No high-cardinality identifier detected.';
  $('outlierPrep').innerHTML=`Method: <b>${esc(p.outlier_method||'IQR clipping')}</b><br><b>${num(p.outlier_values_detected||0)}</b> outlier values detected.`;
  const removed=p.constant_columns_removed||[];
  $('selectionPrep').innerHTML=`Method: <b>${esc(p.feature_selection_method||'Remove constant/zero-variance columns')}</b><br>Removed: <b>${removed.length}</b><br><small>${esc(removed.join(', ')||'None')}</small>`;
  const dist=p.class_distribution||[];
  $('classPrep').innerHTML=dist.length?dist.map(x=>`<div class="mini-dist"><span>${esc(x.class)}</span><b>${num(x.count)}</b></div>`).join(''):'Select a target to view distribution.';
}

async function removeDataset(){
  if(!ds)return;
  if(!confirm('Remove the uploaded CSV and reset the app?'))return;
  await fetch('/api/remove',{method:'POST'});
  ds=null; cols=[]; lastImportance=[]; lastRules=[]; comparisonLoaded=false; modelReady=false;
  $('file').value=''; $('status').textContent='No dataset'; $('rows').textContent='—'; $('features').textContent='—'; $('classes').textContent='—'; $('acc').textContent='—';
  $('target').innerHTML=''; $('info').innerHTML=''; $('prepSummary').innerHTML=''; $('preview').innerHTML='';
  ['missingPrep','duplicatePrep','encodingPrep','idPrep','outlierPrep','selectionPrep','classPrep'].forEach(id=>$(id).textContent='Upload a dataset.');
  $('removeFile').disabled=true; $('trainmsg').textContent=''; show('dataset');
}

$('file').onchange = async () => {
  const f=$('file').files[0]; if(!f)return;
  const fd=new FormData(); fd.append('file',f);
  const r=await fetch('/api/upload',{method:'POST',body:fd}); const d=await r.json();
  if(!r.ok)return alert(d.error);
  ds=d; cols=d.columns; renderPreprocessing(d); $('removeFile').disabled=false;
  $('status').textContent=d.filename; $('rows').textContent=num(d.rows); $('features').textContent=Math.max(0,cols.length-(d.ignored_id_columns||[]).length-1);
  $('target').innerHTML=cols.map(c=>`<option value="${esc(c.name)}">${esc(c.name)}</option>`).join(''); $('target').value=cols.at(-1).name;
  $('info').innerHTML=`<p><b>${num(d.rows)}</b> usable rows • <b>${cols.length}</b> columns • <b>${d.numeric_columns}</b> numeric • <b>${d.categorical_columns}</b> categorical</p>`;
  $('prepSummary').innerHTML=`<div><b>Original rows</b><span>${num(d.original_rows)}</span></div><div><b>Missing values</b><span>${num(d.missing_values)}</span></div><div><b>Duplicate rows</b><span>${num(d.duplicate_rows)}</span></div><div><b>Ignored ID columns</b><span>${(d.ignored_id_columns||[]).length?esc(d.ignored_id_columns.join(', ')):'None'}</span></div>`;
  const ks=Object.keys(d.preview[0]||{});
  $('preview').innerHTML='<tr>'+ks.map(k=>`<th>${esc(k)}</th>`).join('')+'</tr>'+d.preview.map(r=>'<tr>'+ks.map(k=>`<td>${esc(r[k])}</td>`).join('')+'</tr>').join('');
  buildForm(); show('train');
};

$('target').onchange=buildForm;
function buildForm(){
  if(!cols.length)return;
  const t=$('target').value; const ids=(ds?.ignored_id_columns||[]);
  $('form').innerHTML=cols.filter(c=>c.name!==t&&!ids.includes(c.name)).map(c=>{
    const type=/int|float|double/.test(c.dtype)?'number':'text';
    return `<label>${esc(c.name)}<input type="${type}" step="any" data-c="${esc(c.name)}"></label>`;
  }).join('');
}

async function train(){
  if(!ds)return alert('Upload a CSV first');
  $('trainmsg').textContent='Training...';
  const body={target:$('target').value,test_size:$('test').value,criterion:$('criterion').value,max_depth:$('depth').value||null,min_samples:$('mins').value};
  const r=await fetch('/api/train',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}),d=await r.json();
  if(!r.ok)return alert(d.error);
  const m=d.metrics;
  $('ma').textContent=pct(m.accuracy); $('mp').textContent=pct(m.precision); $('mr').textContent=pct(m.recall); $('mf').textContent=pct(m.f1);
  $('acc').textContent=pct(m.accuracy); $('classes').textContent=m.classes.length; renderPredictionAccuracy(m.accuracy); $('trainmsg').textContent=`Training complete • ${num(m.train_rows)} train • ${num(m.test_rows)} test`;
  if(d.recategorization?.enabled){ $('trainmsg').textContent += ` • target re-categorized into ${d.recategorization.classes} ranges (${d.recategorization.original_classes} original values)`; }
  lastImportance=d.importances||[]; lastRules=d.rules||[]; modelReady=true;
  $('ruleSource').textContent=`${d.rule_source}. Confidence = how often the rule's predicted class is correct within its leaf; Support = share of training rows covered by that rule.`;
  $('rulesList').innerHTML=d.rules.map(x=>`<div class="rule"><div class="rule-text">IF ${esc(x.rule)}</div><div class="tag">THEN ${esc(x.prediction)}</div><div class="rule-metric"><b>${pct(x.confidence)}</b><small>confidence</small></div><div class="rule-metric"><b>${pct(x.support)}</b><small>support</small></div></div>`).join('');
  $('importance').innerHTML=d.importances.map(x=>`<div class="importance-row"><div class="importance-label"><span>${esc(x.feature)}</span><b>${(x.importance*100).toFixed(2)}%</b></div><div class="bar"><i style="width:${Math.min(100,x.importance*100)}%"></i></div></div>`).join('');
  $('eval').innerHTML=[['Accuracy',m.accuracy],['Macro Precision',m.precision],['Macro Recall',m.recall],['Macro F1',m.f1]].map(x=>`<div class="metric"><span>${x[0]}</span><b>${pct(x[1])}</b></div>`).join('');
  let h='<table><tr><th>Actual / Predicted</th>'+m.classes.map(c=>`<th>${esc(c)}</th>`).join('')+'</tr>';
  m.confusion_matrix.forEach((row,i)=>h+=`<tr><th>${esc(m.classes[i])}</th>${row.map(v=>`<td>${v}</td>`).join('')}</tr>`);
  $('matrix').innerHTML=h+'</table>';
  buildForm();
  await loadVisualData();
  show('evaluation');
}

async function runVisualization(){
  if(!ds){ alert('Upload a dataset first.'); return; }
  if(!modelReady){ alert('Train the Decision Tree first, then run visualization from this tab.'); return; }
  $('vizStatus').textContent='Building dataset visualizations...';
  const button=document.querySelector('.viz-actions button');
  if(button) button.disabled=true;
  try{
    await loadVisualData();
    $('vizStatus').textContent='Running Decision Tree comparisons...';
    await runComparison();
    $('vizStatus').textContent='Visualization complete.';
  }catch(e){
    $('vizStatus').textContent=`Visualization failed: ${e.message}`;
  }finally{
    if(button) button.disabled=false;
  }
}

async function runComparison(){
  if(!ds)return alert('Upload a CSV first');
  $('vizStatus').textContent='Running Decision Tree comparisons. Large datasets use a representative sample for speed...';
  const body={target:$('target').value,current_criterion:$('criterion').value,current_test_size:$('test').value,current_depth:$('depth').value||null,min_samples:$('mins').value};
  const r=await fetch('/api/compare',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}),d=await r.json();
  if(!r.ok){$('vizStatus').textContent='Comparison failed';return alert(d.error)}
  comparisonLoaded=true; $('vizStatus').textContent=`Comparison ready${d.sampled ? ` • compared on ${num(d.comparison_rows)} representative rows for speed` : ''}.`;
  renderMetricChart('criterionChart',d.criteria,'Criterion');
  renderMetricChart('testChart',d.test_sizes,'Test size');
  renderMetricChart('depthChart',d.depths,'Max depth');
  renderSingleBar('classChart',d.class_distribution||[],'count',false);
}

function renderMetricChart(id,data,title){
  const metrics=[['accuracy','Accuracy'],['precision','Precision'],['recall','Recall'],['f1','F1']];
  $(id).innerHTML=data.map(row=>`<div class="chart-group"><div class="chart-name">${esc(row.name)}</div>${metrics.map(([k,l])=>`<div class="chart-line"><span>${l}</span><div class="track"><i style="width:${Math.max(2,row[k]*100)}%"></i></div><b>${pct(row[k])}</b></div>`).join('')}</div>`).join('');
}
function renderPredictionAccuracy(value){
  const v=Math.max(0,Math.min(1,Number(value)||0));
  const pctValue=(v*100).toFixed(1);
  $('predictionAccuracyChart').innerHTML=`<div class="single-chart-row accuracy-row"><span>Test accuracy</span><div class="track"><i style="width:${Math.max(2,v*100)}%"></i></div><b>${pctValue}%</b></div>`;
}

function renderSingleBar(id,data,valueKey,percentage,suffix=''){
  const max=Math.max(...data.map(x=>Number(x[valueKey])||0),1);
  $(id).innerHTML=data.map(row=>{const v=Number(row[valueKey])||0;return `<div class="single-chart-row"><span>${esc(row.name)}</span><div class="track"><i style="width:${Math.max(2,v/max*100)}%"></i></div><b>${percentage?v.toFixed(2)+suffix:num(v)}</b></div>`}).join('');
}

async function loadVisualData(){
  if(!ds)return;
  $('histogramCharts').innerHTML='<p class="muted">Building histograms...</p>';
  $('categoryCharts').innerHTML='<p class="muted">Building bar graphs...</p>';
  try{
    const r=await fetch('/api/visual-data',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({target:$('target').value})});
    const d=await r.json();
    if(!r.ok) throw new Error(d.error||'Visualization failed');
    $('histogramCharts').innerHTML=(d.histograms||[]).map(h=>renderHistogram(h)).join('') || '<p class="muted">No numeric features available.</p>';
    $('categoryCharts').innerHTML=(d.categories||[]).map(c=>`<div class="mini-chart"><h4>${esc(c.name)}</h4>${renderBars(c.values,'count',false)}</div>`).join('') || '<p class="muted">No categorical features available.</p>';
  }catch(e){
    $('histogramCharts').innerHTML=`<p class="error">${esc(e.message)}</p>`;
    $('categoryCharts').innerHTML=`<p class="error">${esc(e.message)}</p>`;
  }
}
function renderBars(data,key,percentage){ const max=Math.max(...data.map(x=>Number(x[key])||0),1); return data.map(x=>{const v=Number(x[key])||0; return `<div class="single-chart-row"><span>${esc(x.name)}</span><div class="track"><i style="width:${Math.max(2,v/max*100)}%"></i></div><b>${percentage?pct(v):num(v)}</b></div>`}).join(''); }
function renderHistogram(h){ const max=Math.max(...h.bins.map(x=>Number(x.count)||0),1); return `<div class="hist-card"><h4>${esc(h.name)}</h4><div class="histogram">${h.bins.map(x=>{const v=Number(x.count)||0; const height=Math.max(4,(v/max)*76); return `<div class="hist-bin" title="${esc(x.label)}: ${num(v)}"><i style="height:${height}%"></i><span>${num(v)}</span></div>`}).join('')}</div><div class="hist-labels"><span>${esc(h.min)}</span><span>${esc(h.max)}</span></div></div>`; }

async function predict(){
  let row={}; document.querySelectorAll('#form input').forEach(i=>row[i.dataset.c]=i.value);
  const r=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({row})}),d=await r.json();
  if(!r.ok)return alert(d.error);
  renderPredictionAccuracy(d.model_accuracy);
  const probabilityNote = d.probability_type === 'calibrated_sigmoid'
    ? 'Calibrated probability • reduces overconfident 0%/100% tree outputs'
    : 'Model probability';
  $('result').innerHTML=`<div class="prediction"><span>Predicted class</span><b>${esc(d.prediction)}</b>${d.confidence!=null?`<small>${probabilityNote}: <b>${d.confidence.toFixed(2)}%</b> • ${d.granularity||0} class probabilities</small>`:''}${d.probabilities?d.probabilities.map(x=>`<div class="prob"><span>${esc(x.class)}</span><div class="track"><i style="width:${x.percentage}%"></i></div><b>${x.percentage.toFixed(2)}%</b></div>`).join(''):''}</div>`;
}

show('overview');
