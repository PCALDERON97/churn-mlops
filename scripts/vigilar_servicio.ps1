# Consulta el servicio cada medio segundo. Dejalo corriendo mientras borrás un pod:
# si nunca aparece FALLA, el servicio no dejó de responder.
# Uso: .\scripts\vigilar_servicio.ps1      (Ctrl+C para cortar)
param([string]$url = "http://localhost:30080/health")
while ($true) {
    $hora = Get-Date -Format "HH:mm:ss.fff"
    try {
        $r = Invoke-RestMethod -Uri $url -DisableKeepAlive -TimeoutSec 2
        "$hora  OK     $($r.pod)"
    } catch {
        "$hora  FALLA  $($_.Exception.Message)"
    }
    Start-Sleep -Milliseconds 500
}
