{ pkgs ? import <nixpkgs> { }, }:
pkgs.mkShell {
  buildInputs = with pkgs; [
    gcc
    python3
    (pkgs.python3.withPackages (python-pkgs: [
      python-pkgs.matplotlib
      python-pkgs.pandas
    ]))
  ];
}
