//-------------------------------------------------------------------------
//   Copyright 2002-2023 National Technology & Engineering Solutions of
//   Sandia, LLC (NTESS).  Under the terms of Contract DE-NA0003525 with
//   NTESS, the U.S. Government retains certain rights in this software.
//
//   This file is part of the Xyce(TM) Parallel Electrical Simulator.
//
//   Xyce(TM) is free software: you can redistribute it and/or modify
//   it under the terms of the GNU General Public License as published by
//   the Free Software Foundation, either version 3 of the License, or
//   (at your option) any later version.
//
//   Xyce(TM) is distributed in the hope that it will be useful,
//   but WITHOUT ANY WARRANTY; without even the implied warranty of
//   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//   GNU General Public License for more details.
//
//   You should have received a copy of the GNU General Public License
//   along with Xyce(TM).
//   If not, see <http://www.gnu.org/licenses/>.
//-------------------------------------------------------------------------

//-------------------------------------------------------------------------
//
// Purpose        : Provide a map-reduce parallel interface for Xyce
//
// Creator        : Ned Bingham, Broccoli, LLC
//
// Creation Date  : 12/7/2023
//
//-------------------------------------------------------------------------
#include <Xyce_config.h>

#include <N_CIR_Xyce.h>
#include <N_CIR_ParallelXyce.h>

#include <N_ERH_Message.h>

#include <N_DEV_Device.h>
#include <N_DEV_DeviceMgr.h>
#include <N_IO_OutputMgr.h>
#include <N_IO_ExtOutInterface.h>
#include <N_DEV_Algorithm.h>
#include <N_DEV_GeneralExternal.h>
#include <N_DEV_VectorComputeInterface.h>
#include <N_PDS_MPI.h>

namespace Xyce {
namespace Circuit {

//-----------------------------------------------------------------------------
// Function      : ParallelSimulator::runSimulation
// Purpose       : Main simulation driver.
// Special Notes : Not private as this is also called from N_DAK_DakotaInterface
// Scope         : public
// Creator       : Eric Keiter, SNL, Parallel Computational Sciences
// Creation Date : 5/27/00
//-----------------------------------------------------------------------------
ParallelSimulator::RunStatus ParallelSimulator::runSimulation()
{
#ifdef Xyce_PARALLEL_MPI
  if (rank() == 0 && size() > 1) {
    Util::Marshal writer;
    writer << (int)WorkerCommand::RUN_SIMULATION;
    Parallel::Broadcast(comm(), writer, 0);
  }
#endif
  bool bsuccess = Simulator::runSimulation();
#ifdef Xyce_PARALLEL_MPI
  if (size() > 1) {
    Parallel::OneReduce(comm(), MPI_LAND, &bsuccess, 1, 0); // assume rank 0 is root
  }
#endif

  return bsuccess ? SUCCESS : ERROR;
}

//-----------------------------------------------------------------------------
// Function      : ParallelSimulator::runWorker
// Purpose       : Worker simulation driver to support interactive commands.
// Special Notes : 
// Scope         : public
// Creator       : Ned Bingham, Broccoli, LLC
// Creation Date : 6/20/23
//-----------------------------------------------------------------------------
ParallelSimulator::RunStatus ParallelSimulator::runWorker()
{
  if (rank() == 0) {
    Report::UserError0() << "runWorker() should not be called from the rank 0 root process.";
    return ERROR;
  }

  while (true) {
    std::string msg;
    Parallel::Broadcast(comm(), msg, 0);
    Util::Marshal reader(msg);

    int command;
    reader >> command;

    if (command == WorkerCommand::RUN_SIMULATION) {
      runSimulation();
    } else if (command == WorkerCommand::SIMULATE_UNTIL) {
      double requestedUntilTime, completedUntilTime;
      reader >> requestedUntilTime;
      simulateUntil(requestedUntilTime, completedUntilTime);
    } else if (command == WorkerCommand::SIMULATION_COMPLETE) {
      simulationComplete();
    } else if (command == WorkerCommand::CHECK_CIRCUIT_PARAMETER_EXISTS) {
      std::string paramName;
      reader >> paramName;
      checkCircuitParameterExists(paramName);
    } else if (command == WorkerCommand::GET_TIME) {
      getTime();
    } else if (command == WorkerCommand::GET_FINAL_TIME) {
      getFinalTime();
    } else if (command == WorkerCommand::GET_DEVICE_NAMES) {
      std::string modelGroup;
      reader >> modelGroup;
      std::vector<std::string> localNames;
      getDeviceNames(modelGroup, localNames);
    } else if (command == WorkerCommand::GET_ALL_DEVICE_NAMES) {
      std::vector<std::string> localNames;
      getAllDeviceNames(localNames);
    } else if (command == WorkerCommand::GET_DAC_DEVICE_NAMES) {
      std::vector<std::string> localNames;
      getDACDeviceNames(localNames);
    } else if (command == WorkerCommand::CHECK_DEVICE_PARAM_NAME) {
      std::string paramName;
      reader >> paramName;
      checkDeviceParamName(paramName);
    } else if (command == WorkerCommand::GET_DEVICE_PARAM_VAL) {
      std::string paramName;
      reader >> paramName;
      double paramValue;
      getDeviceParamVal(paramName, paramValue);
    } else if (command == WorkerCommand::GET_NUM_ADJ_NODES_FOR_DEVICE) {
      std::string deviceName;
      reader >> deviceName;
      int adjNodes = 0;
      getNumAdjNodesForDevice(deviceName, adjNodes);
    } else if (command == WorkerCommand::GET_ADJ_GIDS_FOR_DEVICE) {
      std::string deviceName;
      reader >> deviceName;
      std::vector<int> gids;
      getAdjGIDsForDevice(deviceName, gids);
    } else if (command == WorkerCommand::GET_ADC_MAP) {
      std::map<std::string, std::map<std::string, double> > ADCMap;
      getADCMap(ADCMap);
    } else if (command == WorkerCommand::UPDATE_TIME_VOLTAGE_PAIRS) {
      std::map<std::string, std::vector<std::pair<double,double> >* > timeVoltageMap;
      reader >> timeVoltageMap;
      updateTimeVoltagePairs(timeVoltageMap);
      for (auto i = timeVoltageMap.begin(); i != timeVoltageMap.end(); i++) {
        delete i->second;
      }
      timeVoltageMap.clear();
    } else if (command == WorkerCommand::GET_TIME_VOLTAGE_PAIRS) {
      Util::Marshal writer;
      std::map<std::string, std::vector<std::pair<double,double> > > timeVoltageMap;
      getTimeVoltagePairs(timeVoltageMap);
    } else if (command == WorkerCommand::GET_TIME_VOLTAGE_PAIRS_SZ) {
      Util::Marshal writer;
      int maximumSize;
      getTimeVoltagePairsSz(maximumSize);
    } else if (command == WorkerCommand::GET_TIME_STATE_PAIRS) {
      Util::Marshal writer;
      std::map<std::string, std::vector<std::pair<double,int> > > timeStateMap;
      getTimeStatePairs(timeStateMap);
    } else if (command == WorkerCommand::SET_ADC_WIDTHS) {
      std::map<std::string, int> ADCWidthMap;
      reader >> ADCWidthMap;
      setADCWidths(ADCWidthMap);
      ADCWidthMap.clear();
    } else if (command == WorkerCommand::GET_ADC_WIDTHS) {
      std::map<std::string, int> ADCWidthMap;
      reader >> ADCWidthMap;
      getADCWidths(ADCWidthMap);
      ADCWidthMap.clear();
    } else if (command == WorkerCommand::GET_CIRCUIT_VALUE) {
      std::string paramName;
      double paramValue;
      reader >> paramName;
      getCircuitValue(paramName, paramValue);
    } else if (command == WorkerCommand::SET_CIRCUIT_PARAMETER) {
      std::string paramName;
      double paramValue;
      reader >> paramName >> paramValue;
      setCircuitParameter(paramName, paramValue);
    } else if (command == WorkerCommand::CHECK_RESPONSE_VAR) {
      std::string paramName;
      reader >> paramName;
      checkResponseVar(paramName);
    } else if (command == WorkerCommand::OBTAIN_RESPONSE) {
      std::string paramName;
      double paramValue;
      reader >> paramName;
      obtainResponse(paramName, paramValue);
    } else if (command == WorkerCommand::FINALIZE) {
      return SUCCESS;
    } else {
      Report::UserError0() << "Unrecognized worker command '" << command << "'.";
      return ERROR;
    }
  }
  return ERROR;
}

// ---------------------------------------------------------------------------
// API METHODS NEEDED FOR MIXED-SIGNAL and other external applications
//
//-----------------------------------------------------------------------------
// Function      : ParallelSimulator::initialize
// Purpose       : capture all "initialization-type" activities in one
//                 method
// Special Notes :
// Scope         : public
// Creator       : Tom Russo, SNL, Component Information and Models
// Creation Date : 02/17/2004
//-----------------------------------------------------------------------------
ParallelSimulator::RunStatus ParallelSimulator::initialize(
  int           argc,
  char **       argv)
{
  ParallelSimulator::RunStatus run_status = SUCCESS;

#ifdef Xyce_PARALLEL_MPI
  if (rank() == 0 and size() > 1) {
    Util::Marshal writer;
    writer << (int)WorkerCommand::INITIALIZE;
    writer << argc;
    for (int i = 0; i < argc; i++) {
      writer << std::string(argv[i]);
    }
    Parallel::Broadcast(comm(), writer, 0);
  }
#endif

  run_status = initializeEarly(argc, argv);
  if (run_status == SUCCESS) {
    run_status = initializeLate();
  }

  bool success = (run_status == SUCCESS);
#ifdef Xyce_PARALLEL_MPI
  if (size() > 1) {
    Parallel::OneReduce(comm(), MPI_LAND, &success, 1, 0);
  }
#endif

  return success ? SUCCESS : ERROR;
}

//-----------------------------------------------------------------------------
// Function      : ParallelSimulator::getDeviceNames
// Purpose       : get all names of devices of specified type (modelGroupName) 
//                 in the netlist
// Special Notes : "modelGroupName" takes a string of the form the devices would
//                 have when instantiated in a netlist, e.g. "R" for resistors or 
//                 "YGENEXT" for GenExt devices.  For U devices, the modelGroupName would
//                 be "BUF" and not UBUF".  The returned device name(s) will be the 
//                 fully qualified name(s), including any subcircuit hierarchy.
//                 For a YADC device named ADC1, it would be YADC!ADC1.
//                 This function will return false if the requested modelGroupName
//                 is invalid.  It will also return false if the netlist does not
//                 contain any devices for a valid modelGroupName.
// Scope         : public
// Creator       : Tom Russo, SNL, Electrical and Microsystems Modeling
// Creation Date : 8/25/08
//-----------------------------------------------------------------------------
bool ParallelSimulator::getDeviceNames(const std::string &modelGroupName, std::vector<std::string> &deviceNames)
{
#ifdef Xyce_PARALLEL_MPI
  if (rank() == 0 && size() > 1) {
    Util::Marshal writer;
    writer << (int)WorkerCommand::GET_DEVICE_NAMES << modelGroupName;
    Parallel::Broadcast(comm(), writer, 0); // assume rank 0 is root
  }
#endif

  std::vector<std::string> local;
  bool bsuccess = Simulator::getDeviceNames(modelGroupName, local);

#ifdef Xyce_PARALLEL_MPI
  if (size() > 1) {
    std::vector<std::string> deviceGroups;
    Util::Marshal writer;
    writer << local;
    Parallel::GatherV(comm(), 0, writer.str(), deviceGroups);
    //MPI_Reduce(&bsuccess, &bsuccess, 1, MPI_CXX_BOOL, MPI_LOR, 0, comm());
    Parallel::OneReduce(comm(), MPI_LOR, &bsuccess, 1, 0); // assume rank 0 is root

    if (rank() == 0) { // assume rank 0 is root
      local.clear();
      for (int i = 0; i < (int)deviceGroups.size(); i++) {
        Util::Marshal reader(deviceGroups[i]);
        reader >> local;
      }
    }
  }
#endif

  deviceNames.insert(deviceNames.end(), local.begin(), local.end());
  return bsuccess;
}

//----------------------------------------------------------------------------
// Function       : ParallelSimulator::getDACDeviceNames
// Purpose        : Gets the fully qualified names of the DAC devices, including
//                  any subcircuit hierarchy.  For a YDAC device named DAC1, it 
//                  would be YDAC!DAC1.
// Special Notes  :
// Scope          :
// Creator        : Lisa Maynes
// Creation Date  : 06/13/2003
//----------------------------------------------------------------------------
bool ParallelSimulator::getDACDeviceNames(std::vector< std::string >& dacNames)
{
#ifdef Xyce_PARALLEL_MPI
  if (rank() == 0 && size() > 1) {
    Util::Marshal writer;
    writer << (int)WorkerCommand::GET_DAC_DEVICE_NAMES;
    Parallel::Broadcast(comm(), writer, 0);
  }
#endif

  std::vector<std::string> local;
  bool bsuccess = Simulator::getDACDeviceNames(local);

#ifdef Xyce_PARALLEL_MPI
  if (size() > 1) {
    std::vector<std::string> deviceGroups;
    Util::Marshal writer;
    writer << local;
    Parallel::GatherV(comm(), 0, writer.str(), deviceGroups);
    Parallel::OneReduce(comm(), MPI_LOR, &bsuccess, 1, 0);

    if (rank() == 0) {
      local.clear();
      for (int i = 0; i < (int)deviceGroups.size(); i++) {
        Util::Marshal reader(deviceGroups[i]);
        reader >> local;
      }
    }
  }
#endif

  dacNames.insert(dacNames.end(), local.begin(), local.end());
  return bsuccess;
}

//-----------------------------------------------------------------------------
// Function      : ParallelSimulator::getAllDeviceNames
// Purpose       : get the names of all devices in the netlist.
// Special Notes : The returned device name(s) will be the fully qualified
//                 name(s), including any subcircuit hierarchy.  This function
//                 will return false if the netlist does not have any devices.
// Scope         : public
// Creator       : Pete Sholander, SNL
// Creation Date : 12/11/2019
//-----------------------------------------------------------------------------
bool ParallelSimulator::getAllDeviceNames(std::vector<std::string> &deviceNames)
{
#ifdef Xyce_PARALLEL_MPI
  if (rank() == 0 && size() > 1) {
    Util::Marshal writer;
    writer << (int)WorkerCommand::GET_ALL_DEVICE_NAMES;
    Parallel::Broadcast(comm(), writer, 0);
  }
#endif

  std::vector<std::string> local;
  bool bsuccess = getAllDeviceNames(local);

#ifdef Xyce_PARALLEL_MPI
  if (size() > 1) {
    std::vector<std::string> deviceGroups;
    Util::Marshal writer;
    writer << local;
    Parallel::GatherV(comm(), 0, writer.str(), deviceGroups);
    Parallel::OneReduce(comm(), MPI_LOR, &bsuccess, 1, 0);

    if (rank() == 0) {
      local.clear();
      for (int i = 0; i < (int)deviceGroups.size(); i++) {
        Util::Marshal reader(deviceGroups[i]);
        reader >> local;
      }
    }
  }
#endif

  deviceNames.insert(deviceNames.end(), local.begin(), local.end());
  return bsuccess;
}

//-----------------------------------------------------------------------------
// Function      : ParallelSimulator::checkDeviceParamName
// Purpose       : check if the specified parameter name (e.g., X1:R1:R) is
//                 a valid parameter for a device that exists in the netlist.
// Special Notes : This function will return false if the device or parameter
//                 does not exist.
// Scope         : public
// Creator       : Pete Sholander, SNL
// Creation Date : 12/18/2019
//-----------------------------------------------------------------------------
bool ParallelSimulator::checkDeviceParamName(const std::string full_param_name) const
{
#ifdef Xyce_PARALLEL_MPI
  if (rank() == 0 && size() > 1) {
    Util::Marshal writer;
    writer << (int)WorkerCommand::CHECK_DEVICE_PARAM_NAME << full_param_name;
    Parallel::Broadcast(comm(), writer, 0);
  }
#endif

  bool bsuccess = Simulator::checkDeviceParamName(full_param_name);

#ifdef Xyce_PARALLEL_MPI
  if (size() > 1) {
    Parallel::OneReduce(comm(), MPI_LOR, &bsuccess, 1, 0);
  }
#endif

  return bsuccess;
}

//-----------------------------------------------------------------------------
// Function      : ParallelSimulator::getDeviceParamVal
// Purpose       : get the value of a specified device parameter.
// Special Notes : This function will return false if the device or parameter
//                 does not exist.
// Scope         : public
// Creator       : Pete Sholander, SNL
// Creation Date : 12/11/2019
//-----------------------------------------------------------------------------
bool ParallelSimulator::getDeviceParamVal(const std::string full_param_name, double& val) const
{
#ifdef Xyce_PARALLEL_MPI
  if (rank() == 0 && size() > 1) {
    Util::Marshal writer;
    writer << (int)WorkerCommand::GET_DEVICE_PARAM_VAL << full_param_name;
    Parallel::Broadcast(comm(), writer, 0);
  }
#endif

  double paramValue = 0.0;
  bool bsuccess = Simulator::getDeviceParamVal(full_param_name, paramValue);

#ifdef Xyce_PARALLEL_MPI
  if (size() > 1) {
    bool found = bsuccess;
    Parallel::OneReduce(comm(), MPI_LOR, &bsuccess, 1, 0);
    if (rank() == 0 and bsuccess and not found) {
      MPI_Status status;
      MPI_Recv(&paramValue, 1, MPI_DOUBLE, MPI_ANY_SOURCE, WorkerCommand::GET_DEVICE_PARAM_VAL, comm(), &status);
    } else if (rank() != 0 and found) {
      MPI_Send(&paramValue, 1, MPI_DOUBLE, 0, WorkerCommand::GET_DEVICE_PARAM_VAL, comm());
    }
  }
#endif

  val = paramValue;
  return bsuccess;
}

//-----------------------------------------------------------------------------
// Function      : ParallelSimulator::getNumAdjNodesForDevice
// Purpose       : get the number of the nodes that are adjacent to a specified
//                 device, including the ground node
// Special Notes : This function will return false if the device does not exist.
// Scope         : public
// Creator       : Pete Sholander, SNL
// Creation Date : 1/7/2020
//-----------------------------------------------------------------------------
bool ParallelSimulator::getNumAdjNodesForDevice(const std::string deviceName, int& numAdjNodes) const
{
#ifdef Xyce_PARALLEL_MPI
  if (rank() == 0 && size() > 1) {
    Util::Marshal writer;
    writer << (int)WorkerCommand::GET_NUM_ADJ_NODES_FOR_DEVICE << deviceName;
    Parallel::Broadcast(comm(), writer, 0);
  }
#endif

  bool retVal = Simulator::getNumAdjNodesForDevice(deviceName, numAdjNodes);

#ifdef Xyce_PARALLEL_MPI
  if (size() > 1) {
    Parallel::OneReduce(comm(), MPI_LOR, &retVal, 1, 0);
    Parallel::OneReduce(comm(), MPI_SUM, &numAdjNodes, 1, 0);
  }
#endif

  return retVal;
}

//-----------------------------------------------------------------------------
// Function      : ParallelSimulator::getAdjGIDsForDevice
// Purpose       : get the GIDs of the nodes that are adjacent to a specified
//                 device, including the ground node (GID of -1).  The other
//                 GIDs are non-negative integers.
// Special Notes : This function will return false if the device does not exist.
// Scope         : public
// Creator       : Pete Sholander, SNL
// Creation Date : 1/7/2020
//-----------------------------------------------------------------------------
bool ParallelSimulator::getAdjGIDsForDevice(const std::string deviceName, std::vector<int> & adj_GIDs) const
{
#ifdef Xyce_PARALLEL_MPI
  if (rank() == 0 && size() > 1) {
    Util::Marshal writer;
    writer << (int)WorkerCommand::GET_ADJ_GIDS_FOR_DEVICE << deviceName;
    Parallel::Broadcast(comm(), writer, 0);
  }
#endif

  std::vector<int> local;
  bool retVal = Simulator::getAdjGIDsForDevice(deviceName, local);

#ifdef Xyce_PARALLEL_MPI
  if (size() > 1) {
    std::vector<std::string> deviceGroups;
    Util::Marshal writer;
    writer << local;
    Parallel::GatherV(comm(), 0, writer.str(), deviceGroups);
    Parallel::OneReduce(comm(), MPI_LOR, &retVal, 1, 0);

    if (rank() == 0) {
      local.clear();
      for (int i = 0; i < (int)deviceGroups.size(); i++) {
        Util::Marshal reader(deviceGroups[i]);
        reader >> local;
      }
    }
  }
#endif

  adj_GIDs.insert(adj_GIDs.end(), local.begin(), local.end());
  return retVal;
}

//----------------------------------------------------------------------------
// Function       : getADCMap
// Purpose        : Gets the names of the ADC devices in the circuit (as the key 
//                  of the map) with an inner map of their instance parameters 
//                  (keyed by parameter name) for each device.  For a YADC 
//                  device named ADC1, the returned name would be YADC!ADC1.
// Special Notes  :
// Scope          :
// Creator        : Tom Russo
// Creation Date  : 05/07/2004
//----------------------------------------------------------------------------
bool ParallelSimulator::getADCMap(std::map<std::string, std::map<std::string, double> >&ADCMap)
{
#ifdef Xyce_PARALLEL_MPI
  if (rank() == 0 && size() > 1) {
    Util::Marshal writer;
    writer << (int)WorkerCommand::GET_ADC_MAP;
    Parallel::Broadcast(comm(), writer, 0);
  }
#endif

  std::map<std::string, std::map<std::string, double> > local;
  bool bsuccess = Simulator::getADCMap(local);

#ifdef Xyce_PARALLEL_MPI
  if (size() > 1) {
    std::vector<std::string> deviceGroups;
    Util::Marshal writer;
    writer << local;
    Parallel::GatherV(comm(), 0, writer.str(), deviceGroups);
    Parallel::OneReduce(comm(), MPI_LOR, &bsuccess, 1, 0);

    if (rank() == 0) {
      local.clear();
      for (int i = 0; i < (int)deviceGroups.size(); i++) {
        std::map<std::string, std::map<std::string, double> > recv;
        Util::Marshal reader(deviceGroups[i]);
        reader >> recv;
        for (auto j = recv.begin(); j != recv.end(); i++) {
          auto loc = local.find(j->first);
          if (loc == local.end()) {
            local.insert(*j);
          } else {
            loc->second.insert(j->second.begin(), j->second.end());
          }
        }
      }
    }
  }
#endif

  ADCMap.insert(local.begin(), local.end());
  return bsuccess;
}

//----------------------------------------------------------------------------
// Function       : updateTimeVoltagePairs
// Purpose        : Update the DAC devices in a circuit by adding the set
//                  of time and voltage pairs built up on the "digital side"
//                  since the last update and by removing the time-voltage
//                  pairs for times that pre-date the given simulation time.
// Special Notes  : The current method for locating DAC instances
//                  works for the serial case only. The parallel
//                  case will be added in later modifications.
// Scope          :
// Creator        : Lon Waters
// Creation Date  : 06/09/2003
//----------------------------------------------------------------------------
bool ParallelSimulator::updateTimeVoltagePairs(const std::map< std::string, std::vector<std::pair<double,double> > *> & timeVoltageUpdateMap)
{
#ifdef Xyce_PARALLEL_MPI
  if (rank() == 0 && size() > 1) {
    Util::Marshal writer;
    writer << (int)WorkerCommand::UPDATE_TIME_VOLTAGE_PAIRS << timeVoltageUpdateMap;
    Parallel::Broadcast(comm(), writer, 0);
  }
#endif

  bool success = Simulator::updateTimeVoltagePairs(timeVoltageUpdateMap);

#ifdef Xyce_PARALLEL_MPI
  if (size() > 1) {
    Parallel::OneReduce(comm(), MPI_LOR, &success, 1, 0);
  }
#endif

  return success;
}

//----------------------------------------------------------------------------
// Function       : getTimeVoltagePairs
// Purpose        : get a map of all time-voltage pairs from all ADC instances
//
// Special Notes  : The current method for locating ADC instances
//                  works for the serial case only. The parallel
//                  case will be added in later modifications.
// Scope          : public
// Creator        : Tom Russo
// Creation Date  : 05/10/2004
//----------------------------------------------------------------------------
bool ParallelSimulator::getTimeVoltagePairs(std::map< std::string, std::vector< std::pair<double,double> > > & timeVoltageUpdateMap)
{
#ifdef Xyce_PARALLEL_MPI
  if (rank() == 0 && size() > 1) {
    Util::Marshal writer;
    writer << (int)WorkerCommand::GET_TIME_VOLTAGE_PAIRS;
    Parallel::Broadcast(comm(), writer, 0);
  }
#endif

  std::map< std::string, std::vector< std::pair<double,double> > > local;
  bool success = Simulator::getTimeVoltagePairs(local);

#ifdef Xyce_PARALLEL_MPI
  if (size() > 1) {
    std::vector<std::string> deviceGroups;
    Util::Marshal writer;
    writer << local;
    Parallel::GatherV(comm(), 0, writer.str(), deviceGroups);
    Parallel::OneReduce(comm(), MPI_LOR, &success, 1, 0);

    if (rank() == 0) {
      local.clear();
      for (int i = 0; i < (int)deviceGroups.size(); i++) {
        std::map< std::string, std::vector< std::pair<double,double> > > recv;
        Util::Marshal reader(deviceGroups[i]);
        reader >> recv;
        for (auto j = recv.begin(); j != recv.end(); j++) {
          auto loc = local.find(j->first);
          if (loc == local.end()) {
            local.insert(*j);
          } else {
            loc->second.insert(loc->second.end(), j->second.begin(), j->second.end());
          }
        }
      }
    }
  }
#endif

  timeVoltageUpdateMap = local;
  return success;
}

//----------------------------------------------------------------------------
// Function       : getTimeVoltagePairsSz
// Purpose        : Returns the largest size of the TV Pairs data in the ADCs
//
// Special Notes  : 
// Creator        : 
// Creation Date  : 11/11/2021
//----------------------------------------------------------------------------
bool ParallelSimulator::getTimeVoltagePairsSz(int &maximumSize)
{
#ifdef Xyce_PARALLEL_MPI
  if (rank() == 0 && size() > 1) {
    Util::Marshal writer;
    writer << (int)WorkerCommand::GET_TIME_VOLTAGE_PAIRS_SZ;
    Parallel::Broadcast(comm(), writer, 0);
  }
#endif

  bool retVal = Simulator::getTimeVoltagePairsSz(maximumSize);

#ifdef Xyce_PARALLEL_MPI
  if (size() > 1) {
    Parallel::OneReduce(comm(), MPI_LOR, &retVal, 1, 0);
    Parallel::OneReduce(comm(), MPI_MAX, &maximumSize, 1, 0);
  }
#endif

  return retVal;
}


//----------------------------------------------------------------------------
// Function       : getTimeStatePairs
// Purpose        : get a map of all time-state pairs from all ADC instances
//
// Special Notes  : The current method for locating ADC instances
//                  works for the serial case only. The parallel
//                  case will be added in later modifications.
// Scope          : public
// Creator        : Pete Sholander
// Creation Date  : 11/13/2018
//----------------------------------------------------------------------------
bool ParallelSimulator::getTimeStatePairs(std::map< std::string, std::vector< std::pair<double,int> > > & timeStateUpdateMap)
{
#ifdef Xyce_PARALLEL_MPI
  if (rank() == 0 && size() > 1) {
    Util::Marshal writer;
    writer << (int)WorkerCommand::GET_TIME_STATE_PAIRS;
    Parallel::Broadcast(comm(), writer, 0);
  }
#endif

  std::map< std::string, std::vector< std::pair<double,int> > > local;
  bool success = Simulator::getTimeStatePairs(local);

#ifdef Xyce_PARALLEL_MPI
  if (size() > 1) {
    std::vector<std::string> deviceGroups;
    Util::Marshal writer;
    writer << local;
    Parallel::GatherV(comm(), 0, writer.str(), deviceGroups);
    Parallel::OneReduce(comm(), MPI_LOR, &success, 1, 0);

    if (rank() == 0) {
      local.clear();
      for (int i = 0; i < (int)deviceGroups.size(); i++) {
        std::map< std::string, std::vector< std::pair<double,int> > > recv;
        Util::Marshal reader(deviceGroups[i]);
        reader >> recv;
        for (auto j = recv.begin(); j != recv.end(); j++) {
          auto loc = local.find(j->first);
          if (loc == local.end()) {
            local.insert(*j);
          } else {
            loc->second.insert(loc->second.end(), j->second.begin(), j->second.end());
          }
        }
      }
    }
  }
#endif

  timeStateUpdateMap = local;
  return success;
}

//----------------------------------------------------------------------------
// Function       : setADCWidths
// Purpose        : Update the ADC devices in a circuit by informing them
//                  of the width of their bitvector output on the
//                  "digital side"
// Special Notes  :
// Scope          :
// Creator        : Tom Russo
// Creation Date  : 05/07/2004
//----------------------------------------------------------------------------
bool ParallelSimulator::setADCWidths(const std::map<std::string, int> &ADCWidthMap)
{
#ifdef Xyce_PARALLEL_MPI
  if (rank() == 0 && size() > 1) {
    Util::Marshal writer;
    writer << (int)WorkerCommand::SET_ADC_WIDTHS << ADCWidthMap;
    Parallel::Broadcast(comm(), writer, 0);
  }
#endif

  bool success = Simulator::setADCWidths(ADCWidthMap);

#ifdef Xyce_PARALLEL_MPI
  if (size() > 1) {
    Parallel::OneReduce(comm(), MPI_LOR, &success, 1, 0);
  }
#endif

  return success;
}

//----------------------------------------------------------------------------
// Function       : getADCWidths
// Purpose        : get the width of the specified ADC devices bitvector output 
//                  on the "digital side"
// Special Notes  :
// Scope          :
// Creator        : Pete Sholander
// Creation Date  : 11/16/2018
//----------------------------------------------------------------------------
bool ParallelSimulator::getADCWidths(std::map<std::string, int> &ADCWidthMap)
{
#ifdef Xyce_PARALLEL_MPI
  if (rank() == 0 && size() > 1) {
    Util::Marshal writer;
    writer << (int)WorkerCommand::GET_ADC_WIDTHS << ADCWidthMap;
    Parallel::Broadcast(comm(), writer, 0);
  }
#endif

  bool success = Simulator::getADCWidths(ADCWidthMap);

#ifdef Xyce_PARALLEL_MPI
  if (size() > 1) {
    std::vector<std::string> deviceGroups;
    Util::Marshal writer;
    writer << ADCWidthMap;
    Parallel::GatherV(comm(), 0, writer.str(), deviceGroups);
    Parallel::OneReduce(comm(), MPI_LOR, &success, 1, 0);

    if (rank() == 0) {
      ADCWidthMap.clear();
      for (int i = 0; i < (int)deviceGroups.size(); i++) {
        std::map< std::string, int > recv;
        Util::Marshal reader(deviceGroups[i]);
        reader >> recv;
        for (auto j = recv.begin(); j != recv.end(); j++) {
          auto loc = ADCWidthMap.find(j->first);
          if (loc == ADCWidthMap.end()) {
            ADCWidthMap.insert(*j);
          } else {
            loc->second = j->second;
          }
        }
      }
    }
  }
#endif

  return success;
}

//---------------------------------------------------------------------------
// Function      : ParallelSimulator::simulateUntil
// Purpose       : To continue the existing analog circuit simulation
//                 until either the given <requestedUntilTime> is reached
//                 or the simulation termination criterion is met.
//                 Return a Boolean indicating whether the simulation
//                 run was successful. (Note that the run is successful
//                 even when the given <requestedUntilTime> is not reached,
//                 so long as the run completed normally.)
// Special Notes : The time variables are in units of seconds.
// Scope         : public
// Creator       : Lon Waters, SNL
//               : Tom Russo, SNL, Component Information and Models
// Creation Date : 06/03/2003
//---------------------------------------------------------------------------
bool ParallelSimulator::simulateUntil(
  double        requestedUntilTime,
  double &      completedUntilTime)
{
#ifdef Xyce_PARALLEL_MPI
  if (rank() == 0 && size() > 1) {
    Util::Marshal writer;
    writer << (int)WorkerCommand::SIMULATE_UNTIL << requestedUntilTime;
    Parallel::Broadcast(comm(), writer, 0);
  }
#endif

  bool bsuccess = Simulator::simulateUntil(requestedUntilTime, completedUntilTime);

#ifdef Xyce_PARALLEL_MPI
  if (size() > 1) {
    Parallel::OneReduce(comm(), MPI_MIN, &completedUntilTime, 1, 0);
    Parallel::OneReduce(comm(), MPI_LAND, &bsuccess, 1, 0);
  }
#endif

  return bsuccess;
}

//---------------------------------------------------------------------------
// Function      : ParallelSimulator::finalize
// Purpose       : To clean up after driving Xyce with the SIMBUS
//                 simulation backplane. This includes the following:
//                    Free any dynamically allocated memory...
// Special Notes :
// Scope         : public
// Creator       : Lon Waters, SNL
//               : Lisa Maynes, CoMeT Solutions, Inc.
// Creation Date : 06/03/2003
//---------------------------------------------------------------------------
ParallelSimulator::RunStatus ParallelSimulator::finalize()
{
#ifdef Xyce_PARALLEL_MPI
  if (rank() == 0 && size() > 1) {
    Util::Marshal writer;
    writer << (int)WorkerCommand::FINALIZE;
    Parallel::Broadcast(comm(), writer, 0);
  }
#endif

  bool success = (Simulator::finalize() == SUCCESS);

#ifdef Xyce_PARALLEL_MPI
  if (size() > 1) {
    Parallel::OneReduce(comm(), MPI_LOR, &success, 1, 0);
  }
#endif

  return success ? SUCCESS : ERROR;
}

//---------------------------------------------------------------------------
// Function      : ParallelSimulator::simulationComplete
// Purpose       : Simply report whether we've reached the end of the
//                 simulation
// Special Notes :
// Scope         : public
// Creator       : Tom Russo, SNL, Component Information and Models
// Creation Date : 05/06/2004
//---------------------------------------------------------------------------
bool ParallelSimulator::simulationComplete()
{
#ifdef Xyce_PARALLEL_MPI
  if (rank() == 0 && size() > 1) {
    Util::Marshal writer;
    writer << (int)WorkerCommand::SIMULATION_COMPLETE;
    Parallel::Broadcast(comm(), writer, 0);
  }
#endif

  bool bcomplete = Simulator::simulationComplete();

#ifdef Xyce_PARALLEL_MPI
  if (size() > 1) {
    Parallel::OneReduce(comm(), MPI_LAND, &bcomplete, 1, 0);
  }
#endif

  return bcomplete;
}

//---------------------------------------------------------------------------
// Function      : ParallelSimulator::checkResponseVar
// Purpose       :
// Special Notes : Used when Dakota or an external program calls Xyce to tell
//                 Xyce what .measure lines are to be used as response functions
//                 to pass back to Dakota.  This call checks the measure manager
//                 in the I/O package has set up measure objects for each label.
// Scope         : public
// Creator       : Heidi Thornquist, Sandia National Labs
// Creation Date : 02/24/2014
//---------------------------------------------------------------------------
bool ParallelSimulator::checkResponseVar(
  const std::string &   variable_name) const
{
#ifdef Xyce_PARALLEL_MPI
  if (rank() == 0 && size() > 1) {
    Util::Marshal writer;
    writer << (int)WorkerCommand::CHECK_RESPONSE_VAR << variable_name;
    Parallel::Broadcast(comm(), writer, 0);
  }
#endif

  bool returnValue = Simulator::checkResponseVar(variable_name);

#ifdef Xyce_PARALLEL_MPI
  if (size() > 1) {
    Parallel::OneReduce(comm(), MPI_LOR, &returnValue, 1, 0);
  }
#endif
  return returnValue;
}

//---------------------------------------------------------------------------
// Function      : ParallelSimulator::obtainResponse
// Purpose       :
// Special Notes : Used when Dakota or an external program calls Xyce to tell
//                 Xyce what .measure lines are to be used as response functions
//                 to pass back to Dakota.  This call obtains the responses for
//                 each labelled measure.
// Scope         : public
// Creator       : Heidi Thornquist, Sandia National Labs
// Creation Date : 02/24/2014
//---------------------------------------------------------------------------
bool ParallelSimulator::obtainResponse(
  const std::string &   variable_name,
  double &              result) const
{
#ifdef Xyce_PARALLEL_MPI
  if (rank() == 0 && size() > 1) {
    Util::Marshal writer;
    writer << (int)WorkerCommand::OBTAIN_RESPONSE << variable_name;
    Parallel::Broadcast(comm(), writer, 0);
  }
#endif

  double paramValue = 0.0;
  bool bsuccess = Simulator::obtainResponse(variable_name, paramValue);

#ifdef Xyce_PARALLEL_MPI
  if (size() > 1) {
    bool found = bsuccess;
    Parallel::OneReduce(comm(), MPI_LOR, &bsuccess, 1, 0);
    if (rank() == 0 and bsuccess and not found) {
      MPI_Status status;
      MPI_Recv(&paramValue, 1, MPI_DOUBLE, MPI_ANY_SOURCE, WorkerCommand::OBTAIN_RESPONSE, comm(), &status);
    } else if (rank() != 0 and found) {
      MPI_Send(&paramValue, 1, MPI_DOUBLE, 0, WorkerCommand::OBTAIN_RESPONSE, comm());
    }
  }
#endif

  result = paramValue;
  return bsuccess;
}


//
// sets the given parameter through the device manager.
// returns false if the parameter did not exist (and thus was not set)
//
bool ParallelSimulator::checkCircuitParameterExists(std::string paramName)
{
#ifdef Xyce_PARALLEL_MPI
  if (rank() == 0 && size() > 1) {
    Util::Marshal writer;
    writer << (int)WorkerCommand::CHECK_CIRCUIT_PARAMETER_EXISTS << paramName;
    Parallel::Broadcast(comm(), writer, 0);
  }
#endif

  bool returnValue = Simulator::checkCircuitParameterExists(paramName);

#ifdef Xyce_PARALLEL_MPI
  if (size() > 1) {
    Parallel::OneReduce(comm(), MPI_LOR, &returnValue, 1, 0);
  }
#endif
  return returnValue;
}

//
// sets the given parameter through the device manager.
// returns false if the parameter did not exist (and thus was not set)
//
bool ParallelSimulator::setCircuitParameter(std::string paramName, double paramValue)
{
#ifdef Xyce_PARALLEL_MPI
  if (rank() == 0 && size() > 1) {
    Util::Marshal writer;
    writer << (int)WorkerCommand::SET_CIRCUIT_PARAMETER << paramName << paramValue;
    Parallel::Broadcast(comm(), writer, 0);
  }
#endif

  bool bsuccess = Simulator::setCircuitParameter(paramName, paramValue);

#ifdef Xyce_PARALLEL_MPI
  if (size() > 1) {
    Parallel::OneReduce(comm(), MPI_LOR, &bsuccess, 1, 0);
  }
#endif
  return bsuccess;
}

//
// gets a value from the current simulation based on the name passed it
// The name can be parameter name, voltage node or current (ie. solution variable)
// or a measure name.  return false if the value was not found.
//
bool ParallelSimulator::getCircuitValue(std::string paramName, double& paramValue)
{
#ifdef Xyce_PARALLEL_MPI
  if (rank() == 0 && size() > 1) {
    Util::Marshal writer;
    writer << (int)WorkerCommand::GET_CIRCUIT_VALUE << paramName;
    Parallel::Broadcast(comm(), writer, 0);
  }
#endif

  double result;
  bool bsuccess = Simulator::getCircuitValue(paramName, result);

#ifdef Xyce_PARALLEL_MPI
  if (size() > 1) {
    bool found = bsuccess;
    Parallel::OneReduce(comm(), MPI_LOR, &bsuccess, 1, 0);
    if (rank() == 0 and bsuccess and not found) {
      MPI_Status status;
      MPI_Recv(&result, 1, MPI_DOUBLE, MPI_ANY_SOURCE, WorkerCommand::GET_CIRCUIT_VALUE, comm(), &status);
    } else if (rank() != 0 and found) {
      MPI_Send(&result, 1, MPI_DOUBLE, 0, WorkerCommand::GET_CIRCUIT_VALUE, comm());
    }
  }
#endif

  paramValue = result;
  return bsuccess;
}

//
// accessor to AnalysisManger funciton
//
double ParallelSimulator::getTime()
{
#ifdef Xyce_PARALLEL_MPI
  if (rank() == 0 && size() > 1) {
    Util::Marshal writer;
    writer << (int)WorkerCommand::GET_TIME;
    Parallel::Broadcast(comm(), writer, 0);
  }
#endif

  double btime = Simulator::getTime();

#ifdef Xyce_PARALLEL_MPI
  if (size() > 1) {
    Parallel::OneReduce(comm(), MPI_MIN, &btime, 1, 0);
  }
#endif
  return btime;
}

//
// accessor to AnalysisManger funciton
//
double ParallelSimulator::getFinalTime()
{
#ifdef Xyce_PARALLEL_MPI
  if (rank() == 0 && size() > 1) {
    Util::Marshal writer;
    writer << (int)WorkerCommand::GET_FINAL_TIME;
    Parallel::Broadcast(comm(), writer, 0);
  }
#endif

  double btime = Simulator::getFinalTime();

#ifdef Xyce_PARALLEL_MPI
  if (size() > 1) {
    Parallel::OneReduce(comm(), MPI_MAX, &btime, 1, 0);
  }
#endif
  return btime;
}


} // namespace Circuit
} // namespace Xyce
