$ErrorActionPreference = "Stop"
.\.venv38\Scripts\Activate.ps1
$babelnetDir = $env:BABELNET_DIR
if (-not $babelnetDir) {
    $babelnetDir = "Path-to-BabelNet"
}
if ($babelnetDir -eq "Path-to-BabelNet") {
    throw "Set BABELNET_DIR before running start_RPC_server.ps1."
}
babelnet-rpc start --bn $babelnetDir --m tcp --tcp 7790 --no-doc
