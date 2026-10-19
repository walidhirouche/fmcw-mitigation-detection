// Online adapter: scientific controls and plots stay in the shared app.js.
let completedDownload = null;
let automaticDownload = null;
$('count').max = 200;
$('open-output').textContent = 'Download HDF5';
$('quit').hidden = true;
$('folder').closest('.row').hidden = true;

function standardDownload(result) {
  const link = document.createElement('a');
  link.href = result.download_url;
  link.download = result.filename;
  document.body.append(link);
  link.click();
  link.remove();
}

poll = async function () {
  try {
    const job = await api('status');
    showJob(job);
    if (job.status === 'running') return;
    clearInterval(pollTimer);
    lockForm(false);
    if (job.status === 'complete') {
      completedDownload = job.result;
      $('stat-size').textContent = `Size: ${fmt(job.result.bytes / 1e6, 2)} MB`;
      $('open-output').hidden = false;
      status('File ready. Your download will start; use Download HDF5 if it does not.', 'success');
      if (automaticDownload !== job.result.download_url) {
        automaticDownload = job.result.download_url;
        standardDownload(job.result);
      }
    } else {
      status(job.message || job.status, job.status === 'error' ? 'error' : '');
    }
  } catch (error) {
    status('Connection lost: ' + error.message + '. Waiting to reconnect…', 'error');
  }
};

$('open-output').onclick = async () => {
  if (!completedDownload) return;
  // Save-as requires a fresh user click. Other browsers use their download UI.
  if (!window.showSaveFilePicker) {
    standardDownload(completedDownload);
    return;
  }
  try {
    const file = await window.showSaveFilePicker({
      suggestedName: completedDownload.filename,
      types: [{description: 'HDF5 dataset', accept: {'application/x-hdf5': ['.h5']}}]
    });
    const response = await fetch(completedDownload.download_url);
    if (!response.ok) throw Error('The file has expired. Generate it again.');
    const output = await file.createWritable();
    await response.body.pipeTo(output);
    status('File saved.', 'success');
  } catch (error) {
    if (error.name === 'AbortError') return;
    standardDownload(completedDownload);
    status('Using your browser download. ' + error.message);
  }
};
