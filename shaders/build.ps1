function BuildAll([string[]] $pathlist, [string] $out_extension)
{
    foreach($path in $pathlist)
    {
        $outpath = [io.path]::ChangeExtension($path, $out_extension)
        Write-Host $path "->" $outpath
        glslc $path -o $outpath
    }
}

BuildAll (Get-ChildItem -Path $PSScriptRoot/*.vert -Recurse) "vso"
BuildAll (Get-ChildItem -Path $PSScriptRoot/*.frag -Recurse) "fso"
