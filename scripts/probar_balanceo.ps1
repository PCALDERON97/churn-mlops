# Manda N peticiones al Service y cuenta qué pod respondió cada una.
# Uso: .\scripts\probar_balanceo.ps1            (20 peticiones)
#      .\scripts\probar_balanceo.ps1 -n 50
param([int]$n = 20, [string]$url = "http://localhost:30080/info")
$conteo = @{}
1..$n | ForEach-Object {
    # -DisableKeepAlive: cada petición abre una conexión nueva, así el Service puede elegir otro pod
    $pod = (Invoke-RestMethod -Uri $url -DisableKeepAlive).pod
    $conteo[$pod] = 1 + [int]$conteo[$pod]
    "{0,3}  {1}" -f $_, $pod
}
"`nPeticiones atendidas por cada pod:"
$conteo.GetEnumerator() | Sort-Object Name | Format-Table @{n="Pod";e={$_.Name}}, @{n="Peticiones";e={$_.Value}} -AutoSize
