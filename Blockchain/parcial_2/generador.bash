#!/bin/bash

set -e

# Nombre del proyecto
PROJECT_DIR="Parcial2_Marketplace"
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

echo "Inicializando proyecto Hardhat..."
npm init -y
npm install --save-dev hardhat @nomicfoundation/hardhat-toolbox dotenv

echo "Creando estructura de Hardhat..."
npx hardhat init <<EOF
y
EOF

echo "Configurando red Ephemery..."
cat >> hardhat.config.js <<'EOF'

require("dotenv").config();

module.exports.networks = {
  ephemery: {
    url: "https://rpc.ephemery.dev",
    accounts: [process.env.PRIVATE_KEY]
}
};
EOF

echo "Agregando contrato base Marketplace.sol..."
mkdir -p contracts
cat > contracts/Marketplace.sol <<'EOF'
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC721/extensions/ERC721URIStorage.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract Marketplace is ERC721URIStorage, Ownable {
    uint256 public nextTokenId;
    uint96 public constant FEE_DENOMINATOR = 1000;

    struct Listing {
        address owner;
        uint96 price;
        bool isSold;
    }

    mapping(uint256 => Listing) public listings;
    mapping(address => uint256) public balances;

    event ItemListed(uint256 tokenId, address seller, uint96 price);
    event ItemSold(uint256 tokenId, address buyer, uint96 price);

    constructor() ERC721("Parcial2NFT", "PNFT") {}

    function mintAndList(string memory _uri, uint96 _price) public {
        uint256 tokenId = nextTokenId++;
        _mint(msg.sender, tokenId);
        _setTokenURI(tokenId, _uri);
        listings[tokenId] = Listing(msg.sender, _price, false);
        emit ItemListed(tokenId, msg.sender, _price);
    }

    function buy(uint256 _tokenId) public payable {
        Listing storage item = listings[_tokenId];
        require(!item.isSold, "Item already sold");
        require(msg.value == item.price, "Incorrect value");

        item.isSold = true;
        balances[item.owner] += msg.value;

        _transfer(item.owner, msg.sender, _tokenId);
        emit ItemSold(_tokenId, msg.sender, item.price);
    }

    function getListing(uint256 _tokenId) public view returns (address, uint96, bool) {
        Listing memory l = listings[_tokenId];
        return (l.owner, l.price, l.isSold);
    }

    function withdraw() public {
        uint256 amount = balances[msg.sender];
        require(amount > 0, "Nothing to withdraw");
        balances[msg.sender] = 0;
        payable(msg.sender).transfer(amount);
    }
}
EOF

echo "Instalando React + Vite frontend..."
npm install ethers@5.7 > /dev/null
npm create vite@latest web_app -- --template react > /dev/null
cd web_app
npm install > /dev/null
cd ..

echo "Agregando frontend base en App.jsx..."
cat > web_app/src/App.jsx <<'EOF'
import { useEffect, useState } from 'react';
import { ethers } from 'ethers';

const contractAddress = import.meta.env.VITE_CONTRACT_ADDRESS;
const abi = []; // debes pegar ABI aquí

function App() {
  const [account, setAccount] = useState(null);

  async function connectWallet() {
    const [acc] = await window.ethereum.request({ method: 'eth_requestAccounts' });
    setAccount(acc);
}

  return (
    <div>
      <h1>Marketplace NFT</h1>
      <button onClick={connectWallet}>Conectar MetaMask</button>
      {account && <p>Conectado: {account}</p>}
    </div>
  );
}

export default App;
EOF

echo "Creando archivo .env..."
cat > .env <<EOF
PRIVATE_KEY=tu_clave_privada
VITE_CONTRACT_ADDRESS=0xTuContrato
VITE_RPC_URL=https://rpc.ephemery.dev
EOF

echo "Agregando README.md..."
cat > README.md <<EOF
# DApp NFT Marketplace - Parcial 2

Este proyecto implementa un mercado de NFTs en Solidity y React.

## Requisitos

- Node.js
- MetaMask
- Cuenta en Ephemery (https://rpc.ephemery.dev)

## Instrucciones

1. Copia tu PRIVATE_KEY en el archivo `.env`
2. Despliega el contrato:
   \`\`\`
   npx hardhat run scripts/deploy.js --network ephemery
   \`\`\`
3. Actualiza VITE_CONTRACT_ADDRESS en `.env`
4. Corre el frontend:
   \`\`\`
   cd web_app
   npm run dev
   \`\`\`

EOF

echo "Agregando Makefile..."
cat > Makefile <<EOF
build:
\tnpx hardhat compile

deploy:
\tnpx hardhat run scripts/deploy.js --network ephemery

frontend:
\tcd web_app && npm run dev
EOF

echo "Agregando ejemplo deploy.js..."
mkdir -p scripts
cat > scripts/deploy.js <<'EOF'
async function main() {
  const Marketplace = await ethers.getContractFactory("Marketplace");
  const marketplace = await Marketplace.deploy();
  await marketplace.deployed();
  console.log("Marketplace deployed to:", marketplace.address);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
EOF

echo "Listo. Estructura creada en: $PROJECT_DIR"
